"""
This file contains code for the GaussianJobDispatcher class. This class is responsible for consuming messages from the nerf-in queue,
running the gaussian splatting job, and publishing the results to the nerf-out queue. The class is initialized with a rabbitmq domain and
a base url for http requests. The class connects to rabbitmq and starts the dispatch service. The class has methods for validating jobs,
parsing jobs, running jobs, acknowledging and publishing messages, and starting the dispatcher. The class also has a method for connecting
to rabbitmq. The class has a logger and a list of threads for running jobs.

TODO: Add dynamic prefetch and multi gpu support. HARD TO IMPLEMENT. Will require modifying training code. Dynamic prefetch fin.
TODO: DONE 7/10/24 Add support for cross-process synchronization to logger or separate logger <- this one
TODO: DONE 7/10/24 Class-ify this file
TODO: Log training and model saving (modify inria code)
TODO: Fix Having to duplicate transforms_train.json to transforms_test.json
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
from uuid import UUID
from queue import Queue
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Optional, Tuple


# Globals
base_url = "http://nerf-worker:5200/"


class GaussianJobDispatcher:
    """
    Job dispatcher that consumes from nerf-in,
    dispatches new training jobs, and publishes to nerf-out. Handles
    list of active threads and a cleanup thread to join finished threads.
    """
    def __init__(self, rabbitmq_domain: str = "rabbitmq", base_url: str = "http://nerf-worker-gaussian:5200/"):
        """
        TODO: Communicate with rabbitmq server on port defined in web-server arguments

        Args:
            rabbitmq_domain (str): _description_
            base_url (str): url used for http requests  
        """
        self.logger = nerf_worker_logger("nerf-worker-dispatcher")
        
        self.base_url = base_url
        self.rabbitmq_domain = rabbitmq_domain
        self.threads: List[threading.Thread] = []
        self.channel = None
        self.connection = None
        self.num_gpus = torch.cuda.device_count()
        self.threads = []
        self.thread_queue = Queue()
        self.cleanup_thread = threading.Thread(target=self.cleanup_finished_threads, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("GaussianJobDispatcher Initialized")
        self.logger.info("base_url: %s", self.base_url)
        self.logger.info(f"Number of available GPUs: {self.num_gpus}")

    def start(self):
        """
        Attempts to start the dispatch service, connecting to rabbitmq.
        Will give up if unable to connect after 2 minutes. Runs
        indefinitely until interrupted. Concurrently runs multiple
        jobs on a 1-per-gpu basis.

        TODO: Modify Training code to allow for dynamic prefetch and multi-gpu support

        Raises: Exception if timeout
        """
        start_time = time.time()
        while time.time() < start_time + 60 * 2:
            try:
                self.connect_to_rabbitmq()
                # self.channel.basic_qos(prefetch_count=self.num_gpus)
                self.channel.basic_qos(prefetch_count=1)
                on_message_callback = functools.partial(self.on_message)
                self.channel.basic_consume(
                    queue='nerf-in',
                    on_message_callback=on_message_callback,
                    auto_ack=False
                )

                try:
                    self.channel.start_consuming()
                except KeyboardInterrupt:
                    self.channel.stop_consuming()
                    self.connection.close()
                    for thread in self.threads:
                        thread.join()
            except pika.exceptions.AMQPConnectionError:
                continue

        self.logger.critical(
            "GaussianJobDispatcher took too long to connect to rabbitmq")
        raise Exception(
            "GaussianJobDispatcher took too long to connect to rabbitmq")

    def shutdown(self):
        """
        Gracefully shuts down the dispatcher. Closes the channel
        and connection to rabbitmq.
        """
        # Signal the cleanup thread to exit
        self.thread_queue.put(None)
        self.cleanup_thread.join()

        # Wait for all worker threads to finish
        for t in self.threads:
            t.join()

        if self.channel and self.channel.is_open:
            self.channel.close()
        if self.connection and self.connection.is_open:
            self.connection.close()

    def connect_to_rabbitmq(self):
        """
        Creates a connection to rabbitmq server
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

    def cleanup_finished_threads(self):
        """
        Cleans up finished threads by joining them and removing from the active threads list.
        """
        while True:
            finished_thread = self.thread_queue.get()
            if finished_thread is None:
                break
            finished_thread.join()
            self.threads = [t for t in self.threads if t.is_alive()]

    def on_message(self, channel, method, header, body):
        """
        Handles new messages from nerf-in consumption. Creates a new 
        thread for the incoming job.

        Args:
            channel (_type_): Pika channel
            method (_type_): Pika method
            header (_type_): Message header
            body (_type_): Message content
            args (_type_): Packed thread list
        """
        self.logger.info("Received message, delivery tag %s", method.delivery_tag)
        self.logger.info("Active threads: %s", len(self.threads))
        delivery_tag = method.delivery_tag

        t = threading.Thread(target=self.run_nerf_job_wrapper, args=(
            channel, method, delivery_tag, body))
        t.start()
        self.threads.append(t)

    def run_nerf_job_wrapper(self, channel, method, delivery_tag, body):
        """
        Wrapper function for running gaussian splatting job. Handles
        thread queue management.

        Args:
            channel (_type_): pika channel
            method (_type_): pika method
            delivery_tag (_type_): message identifier
            body (_type_): job data and configuration
        """
        try:
            self.run_nerf_job(channel, method, delivery_tag, body)
        finally:
            self.thread_queue.put(threading.current_thread())

    def run_nerf_job(self, channel, method, delivery_tag, body):
        """
        Handles the actual gaussian splatting job. Runs training
        with dynamic config. Handles outputs generation and publishes
        appropriate GET endpoints to retrieve the resources specified
        by config.

        FUTURE WORK:
            you MUST use these local channels and connection objects to interact with rabbitMQ
            do NOT use self.blahblah to interact with rabbitMQ
        
        TODO: DONE 7/10/24 Allow user defined snapshot save frequency (default [7000, 30000] iters)
        TODO: DONE 7/9/24 Modify training output to save snapshot point clouds as "UUID.ply" instead of "point_cloud.ply" 

        Args:
            channel (_type_): pika channel
            method (_type_): pika method
            delivery_tag (_type_): message identifier
            body (_type_): job data and configuration
        """
        try:
            # Read job details, convert transforms, and parse into fields.
            job_data = json.loads(body.decode())
            job_data_converted = nerf_utils.convert_transforms_to_gaussian(
                job_data)
            id, output_types, save_iterations, total_iterations, error =\
                self.parse_job(job_data_converted)

            # Invalid job configuration
            if error:
                self.ack_publish_message(channel, delivery_tag, json.dumps({
                    "id": id if id else "None",
                    "error": error
                }))
                return

            # Retrieve images from sfm
            input_dir: Path = Path("data/sfm") / id
            output_dir: Path = Path("data/nerf") / id
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            for i, fr in enumerate(job_data_converted["frames"]):
                url = fr["file_path"]
                img = requests.get(url)
                fr["file_path"] = f"{i}.png"
                img_file_path = input_dir / fr["file_path"]
                img_file_path.write_bytes(img.content)

            input_train = input_dir / "transforms_train.json"
            input_test = input_dir / "transforms_test.json"
            input_train.write_text(json.dumps(job_data_converted, indent=4))
            input_test.write_text(json.dumps(job_data_converted, indent=4))
            self.logger.info(f"Running nerf job for {id}, delivery tag {delivery_tag}")
            self.logger.info(f"Input directory: {input_dir}")

            # Train gaussian splatting model
            train_args = [
                "--source_path", str(input_dir),
                "--model_path", str(output_dir),
                "--save_iterations", *map(str, save_iterations),
                "--iterations", f"{total_iterations}",
                "--job_id", id
            ]
            self.logger.info(f"Train_args {train_args}")
            
            train.main(train_args)

            # Generate Outputs and Populate output dictionary
            output_kwargs = {
                "save_iterations": save_iterations,
                "resolution_width": 480
            }
            output_dict = {
                "id": id,
                "output_endpoints": {}
            }
            output_utils.populate_outputs(
                output_dict, output_types, self.base_url, output_dir, **output_kwargs)

            # Explicitly free GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            # Job Finished. Publish to out queue
            ack_callback = functools.partial(
                self.ack_publish_message, channel, delivery_tag, json.dumps(output_dict))
            channel.connection.add_callback_threadsafe(ack_callback)
            
        except Exception as e:
            error_dict = {
                "id": id if id else "None",
                "error": f"Error in GaussianJobDispatcher: {e}"
            }
            self.logger.exception("Error occurred during training: %s", error_dict["error"])

            ack_callback = functools.partial(
                self.ack_publish_message, channel, delivery_tag, json.dumps(error_dict))
            channel.connection.add_callback_threadsafe(ack_callback)

    def validate_job(self, id, output_types, save_iterations, total_iterations) -> Optional[str]:
        """
        Validates parameters parsed from incoming job.

        TODO: DONE 7/10/24 Implement this

        Args:
            id (_type_): str(uuid4)
            output_types (_type_): list[str], valid entries are "splat", "ply", "video"
            save_iterations (_type_): list[int], each entry must be 0 < entry < total_iterations
            total_iterations (_type_): int, must be below certain
        Returns: Optional[str]. Error message or None for valid inputs
        """

        try:
            uuid = UUID(id, version=4)
        except ValueError:
            return f"Invalid id provided. Job {id}"

        for type in output_types:
            if type not in ["splat_cloud", "point_cloud", "video"]:
                return f"Invalid output type {type} provided. Job {id}"

        if total_iterations > 30000:
            return f"Invalid total_iterations {total_iterations} provided. Job {id}"

        for iteration in save_iterations:
            if iteration < 0 or iteration > total_iterations:
                return f"Invalid save iteration {save_iterations} provided. Job {id}"

        return None

    def parse_job(self, body):
        """
        Parses a incoming nerf job. Attempts to read "id", "output_types", "save_iterations",
        "total_iterations" fields. Field "id" is required. Defaults to "output_types" = ["splat_cloud"],
        "save_iterations" = [7000, 30000], "iterations" = 30000 if not able to be read.

        TODO: DONE 7/10/24 Verifying input integrity, i.e save_iter [9000] and total_iter 300 is wrong

        Args:
            body (_type_): JSON containing job details
        Returns:
            Tuple[Optional[str], Optional[list[str]], Optional[list[int]], Optional[int], Optional[str]]: Tuple of fields
        """

        id = body["id"]
        if not id or not isinstance(id, str):
            error = "No id provided for incoming job, cancelling job"
            self.logger.warning(error)
            return None, None, None, None, error

        output_types = body["output_types"]
        if not output_types:
            self.logger.info(
                "No output types provided, using \"splat_cloud\". Job %s", id)
            output_types = ["splat_cloud"]

        save_iterations = [int(x) for x in body["save_iterations"]]
        if not save_iterations:
            self.logger.info(
                "No save iterations provided, using [7000, 30000]. Job %s", id)
            save_iterations = [7000, 30000]

        total_iterations = int(body["total_iterations"])
        if not total_iterations:
            self.logger.info(
                "No training iterations provided, using 30000. Job %s", id)
            total_iterations = 30000

        error = self.validate_job(
            id, output_types, save_iterations, total_iterations)
        if error:
            error = (
                f"Invalid job fields provided. Error: {error}."
                f"Expected body(id: str(uuid4), output_types: list[\"splat_cloud\"|\"point_cloud\"|\"video \"],"
                f"list[0 < int < total_iterations], total_iterations: int <= 30000). Job {id}"
            )
            self.logger.warning(error)
            return id, None, None, None, error

        return id, output_types, save_iterations, total_iterations, None

    def ack_publish_message(self, channel, delivery_tag, body):
        """
        Threadsafe function to acknowledge received message
        from nerf-in, and publishing to nerf-out if job was
        completed or error.

        Args:
            channel (_type_): Pika channel
            delivery_tag (_type_): Identifier for the message consumed
            body (_type_): Output from running nerf job
        """
        self.logger.info("Publishing message %s, delivery tag %s", body, delivery_tag)

        try:
            self.logger.info("Trying to ack pub body %s", body)
            channel.basic_ack(delivery_tag=delivery_tag)
            if body:
                channel.basic_publish(
                    exchange='', routing_key='nerf-out', body=body)
        except Exception as e:
            self.logger.warning(
                "Failed to ack or publish message. Exception: %s", e)