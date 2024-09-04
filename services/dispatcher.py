"""
This file contains code for the GaussianJobDispatcher class. This class is responsible for consuming messages from the nerf-in queue,
running the gaussian splatting job, and publishing the results to the nerf-out queue. It is capable of running multiple jobs concurrently.

TODO: Add dynamic prefetch and multi gpu support. HARD TO IMPLEMENT. Will require modifying training code. Dynamic prefetch fin.
TODO: DONE 7/10/24 Add support for cross-process synchronization to logger or separate logger <- this one
TODO: DONE 7/10/24 Class-ify this file
TODO: DONEISH 8/22/24 Log training and model saving (modify inria code)
TODO: Fix Having to duplicate transforms_train.json to transforms_test.json (gotta modify inria code again)
"""


import json
import functools
import threading
import pika
import pika.exceptions
import torch
import requests
import train
import time
import os

from log import nerf_worker_logger
from utils import nerf_utils, output_utils

from queue import Queue
from pathlib import Path
from dotenv import load_dotenv


class GaussianJobDispatcher:
    """
    Job dispatcher that consumes from `nerf-in`,
    dispatches new training jobs, and publishes to `nerf-out`. Handles
    lists of active and finished threads and a runs a cleanup thread.
    Is capable of running multiple jobs concurrently on a 1-per-gpu basis.
    """

    def __init__(self, rabbitmq_domain: str = "rabbitmq", base_url: str = "http://nerf-worker:5200/"):
        """
        Args:
            rabbitmq_domain (str): domain for rabbitmq server
            base_url (str): url used for http requests  
        """
        self.logger = nerf_worker_logger("nerf-worker-dispatcher")

        # Rabbitmq and http setup
        self.base_url = base_url
        self.rabbitmq_domain = rabbitmq_domain
        self.connection: pika.BlockingConnection = None
        self.channel = None
        
        # Job and available gpu management
        self.num_gpus = torch.cuda.device_count()
        self.available_gpus = [True] * self.num_gpus # bool for each gpu, True if available, False if in use
        self.threads = []
        
        # Job thread management
        self.finished_thread_queue = Queue()
        self.cleanup_thread = threading.Thread(target=self.cleanup_finished_threads, daemon=True)
        self.cleanup_thread.start()

        self.logger.info("GaussianJobDispatcher Initialized")
        self.logger.info(f"Number of available GPUs: {self.num_gpus}")
        self.logger.debug("base_url: %s", self.base_url)

    def start(self):
        """
        Attempts to start the dispatch service, connecting to rabbitmq.
        Runs indefinitely until interrupted. Concurrently runs multiple
        jobs on a 1-per-gpu basis.

        Raises: Exception if timeout
        """
        start_time = time.time()
        while time.time() < start_time + 60 * 2:
            try:
                self.connect_to_rabbitmq()
                callback = functools.partial(self.on_message, args=(self.connection))
                self.channel.basic_consume(
                    queue='nerf-in',
                    on_message_callback=callback,
                )
                try:
                    self.channel.start_consuming()
                except KeyboardInterrupt:
                    self.shutdown()
                break
            except pika.exceptions.AMQPConnectionError:
                continue

        if time.time() >= start_time + 60 * 2:
            self.logger.critical("GaussianJobDispatcher took too long to connect to rabbitmq")
            raise Exception("GaussianJobDispatcher took too long to connect to rabbitmq")

    def connect_to_rabbitmq(self):
        """
        Creates a blocking connection to rabbitmq server on main thread.
        """
        load_dotenv()
        credentials = pika.PlainCredentials(
            str(os.getenv("RABBITMQ_DEFAULT_USER")),
            str(os.getenv("RABBITMQ_DEFAULT_PASS"))
        )
        parameters = pika.ConnectionParameters(
            self.rabbitmq_domain, 5672, '/', credentials, heartbeat=300
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='nerf-in')
        self.channel.queue_declare(queue='nerf-out')
        self.channel.basic_qos(prefetch_count=1)

    def on_message(self, channel, method, properties, body, args):
        """
        Handles new messages from nerf-in consumption. Creates a new 
        thread for the incoming job.

        Args:
            channel (_type_): Pika channel
            method (_type_): Pika method
            header (_type_): Message header
            body (_type_): Message content
            args: Contains connection
        """
        self.logger.info("Received message, delivery tag %s", method.delivery_tag)
        self.logger.info("Active threads: %s", len(self.threads))
        
        connection = args
        delivery_tag = method.delivery_tag
        
        gpu = self.get_available_gpu()
        if gpu is not None:
            t = threading.Thread(target=self.handle_train_scene, args=(connection, channel, method.delivery_tag, body, gpu))
            t.start()
            self.threads.append(t)
        else:
            self.logger.warning("No available GPUs. Requeuing message.")
            channel.basic_nack(delivery_tag=delivery_tag, requeue=True)

    def get_available_gpu(self):
        """
        Returns:
            (int | None): Index of available gpu, or None if no gpus available
        """
        for i, available in enumerate(self.available_gpus):
            if available:
                self.available_gpus[i] = False
                return i
        return None

    def shutdown(self):
        """
        Gracefully shuts down the dispatcher. Closes the channel
        and connection to rabbitmq. Cleans up all threads.
        """
        self.finished_thread_queue.put(None)
        self.cleanup_thread.join()

        for t in self.threads:
            t.join()

        if self.channel and self.channel.is_open:
            self.channel.close()
        if self.connection and self.connection.is_open:
            self.connection.close()
    
    def cleanup_finished_threads(self):
        """
        Cleans up finished threads by joining them and removing from the active threads list.
        """
        while True:
            finished_thread = self.finished_thread_queue.get()
            if finished_thread is None:
                break
            finished_thread.join()
            self.threads = [t for t in self.threads if t.is_alive()]

    def handle_finished_job(self, channel, delivery_tag, body, gpu_id):
        """
        Threadsafe function to acknowledge received message
        from nerf-in, and publishing to nerf-out if job was
        completed or error. This should be called from the thread
        that the job was dispatched to.

        Args:
            channel (_type_): Pika channel
            delivery_tag (_type_): Identifier for the message consumed
            body (_type_): Output from running nerf job
            gpu_id (int): Index of the gpu used for the job
        """
        self.logger.info("Publishing message %s, delivery tag %s", body, delivery_tag)

        try:
            self.logger.info("Trying to ack pub body %s", body)
            if body:
                channel.basic_publish(exchange='', routing_key='nerf-out', body=body)
            channel.basic_ack(delivery_tag)
            
            self.available_gpus[gpu_id] = True
            
        except Exception as e:
            self.logger.warning("Failed to ack or publish message. Exception: %s", e)

    def handle_train_scene(self, connection, channel, delivery_tag, body, gpu_id):
        """
        Handles the actual gaussian splatting job. Runs training
        with dynamic config. Handles output generation and publishes
        appropriate GET endpoints to retrieve the resources specified
        by config.
        
        Aside:
            you MUST use the passed local connection and channel, as well as 
            threadsafe callback to interact with rabbitmq. There are specific 
            identifiers for a channel and its host thread, and using a different
            channel will cause the ack to fail.
        Args:
            connection (_type_): Pika connection
            channel (_type_): Pika channel
            delivery_tag (_type_): Identifier for the message consumed
            body (_type_): Message content
            gpu_id (int): Index of the gpu to run the
        """
        output = {}
        try:
            torch.cuda.set_device(gpu_id)
            
            # Read job details, convert transforms, and parse into fields.
            self.logger.info("Received job data: %s", body)
            
            job_data = json.loads(body.decode())
            job_data_converted = nerf_utils.convert_transforms_to_gaussian(job_data)
            
            # print('Job data converted: ', job_data_converted, flush=True)

            # Dynamic config parsing
            scene_id = job_data_converted["id"]
            frames = job_data_converted["frames"]
            output_types = job_data_converted["output_types"]
            save_iterations = job_data_converted["save_iterations"]
            total_iterations = job_data_converted["total_iterations"]
            vid_width = job_data_converted["vid_width"]
            white_background = bool(job_data_converted["white_background"])

            self.logger.debug("white_background: %s", white_background)

            # Retrieve images from sfm
            input_dir: Path = Path("data/sfm") / scene_id
            output_dir: Path = Path("data/nerf") / scene_id
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            for i, fr in enumerate(frames):
                self.logger.debug("Downloading image %s", fr["file_path"])
                img = requests.get(fr["file_path"])

                fr["file_path"] = f"{i}.png"
                img_file_path = input_dir / fr["file_path"]
                img_file_path.write_bytes(img.content)

            input_train = input_dir / "transforms_train.json"
            input_test = input_dir / "transforms_test.json"
            input_train.write_text(json.dumps(job_data_converted, indent=4))
            input_test.write_text(json.dumps(job_data_converted, indent=4))

            self.logger.debug(f"Running nerf job for {scene_id}, delivery tag {delivery_tag}")
            self.logger.debug(f"Input directory: {input_dir}")

            # Train gaussian splatting model
            train_args = [
                "--source_path", str(input_dir),
                "--model_path", str(output_dir),
                "--save_iterations", *map(str, save_iterations),
                "--iterations", f"{total_iterations}",
                "--job_id",  scene_id
            ]
            if white_background:
                train_args.append("-w")
            
            self.logger.debug(f"Train_args {train_args}")

            # Run training
            train.main(train_args)

            # Generate Outputs and Populate output dictionary
            output_dict = {}
            output_dict["file_paths"] = output_utils.populate_outputs(self.base_url, output_dir, scene_id,  output_types, save_iterations, vid_width)
            output_dict["id"] = scene_id

            # Explicitly free GPU memory in case research code bad
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Job Finished. Publish to out queue
            output = json.dumps(output_dict)
            self.logger.debug("Job finished, output: %s", output)
            
        except Exception as e:
            self.logger.exception("Error occurred during training: %s", e)
            output = json.dumps({
                "id": scene_id if scene_id else "None",
                "error": f"Error in GaussianJobDispatcher: {e}"
            })
            
        finally:
            self.finished_thread_queue.put(threading.current_thread())                
            ack_callback = functools.partial(self.handle_finished_job, channel, delivery_tag, output, gpu_id)
            connection.add_callback_threadsafe(ack_callback)
