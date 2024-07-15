"""
This module contains code for RabbitMQService class which is responsible for communicating with 
the RabbitMQ server to publish new jobs to the workers. It also contains code for the individual thread
functions that handle the processing of completed jobs from the RabbitMQ server.
"""

import json
import threading
from typing_extensions import deprecated
import pika, os, logging
import requests
import time
import os
import math
import random
import sklearn.cluster
import logging
import numpy as np

from models.scene import NerfV2, Video, Sfm, Nerf, TrainingConfig
from models.managers import SceneManager, QueueListManager

from pathlib import Path
from urllib.parse import urlparse
from flask import url_for, Response
from dotenv import load_dotenv


# Load environment variables from .env file at the root of the project
load_dotenv()

# TODO: Treat this like a singleton, dont want a bajillion instances and 2 bajillion consumers
class RabbitMQServiceV2:
    """
    Service to handle RabbitMQ connections and publish to and consume from the workers.
    For each finished job stage, create a consumer thread with its own connection
    to pull the finished jobs from respective queue.
    
    TODO: make rabbitmq resistent to failed worker jobs
    """
    def __init__(self, rabbitip: str, manager: QueueListManager, scene_manager: SceneManager):
        self.logger = logging.getLogger('web-server')
        self.rabbitmq_domain = rabbitip
        self.queue_manager = manager
        self.scene_manager = scene_manager
        self.base_url = "http://host.docker.internal:5000/"
        self.credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
        self.parameters = pika.ConnectionParameters(self.rabbitmq_domain, 5672, '/', self.credentials, heartbeat=300)
        # Publish function connections
        self.connect()
        
        # Start consumer threads
        self.start_consumers()

    def connect(self):
        """
        Creates publisher connection to RabbitMQ and declares the queues.

        Raises:
            Exception: If the connection takes too long to establish
        """
        timeout = time.time() + 60 * 2
        while True:
            if time.time() > timeout:
                self.logger.critical("RabbitMQService took too long to connect publisher (main) thread to rabbitmq")
                raise Exception("RabbitMQService took too long to connect to rabbitmq")
            try:
                self.connection = pika.BlockingConnection(self.parameters)  
                self.channel = self.connection.channel() 
                self.channel.queue_declare(queue='sfm-in')
                self.channel.queue_declare(queue='nerf-in')
                self.channel.queue_declare(queue='sfm-out')
                self.channel.queue_declare(queue='nerf-out')
                break
            except pika.exceptions.AMQPConnectionError:
                continue

    def to_url(self, file_path):
        """
        Converts a file path to a URL for the worker to download the file from the web server.
        """
        return self.base_url + "/worker-data/" + file_path

    def publish_sfm_job(self, id: str, vid: Video, config: TrainingConfig):
        """
        Publishes a new job to the sfm-in queue hosted on RabbitMQ.

        Args:
            id (str): job id
            vid (Video): video object
            config (TrainingConfig): job specific training config object
        """
        job = {
            "id": id,
            "file_path": self.to_url(vid.file_path)
        }
        
        # Merge specific config into job details
        job = {**job, **config.sfm_config}
        json_job = json.dumps(job)
        self.channel.basic_publish(exchange='', routing_key='sfm-in', body=json_job)
        
        # add to sfm_list and queue_list (first received, goes into overarching queue) queue manager
        self.queue_manager.append_queue("sfm_list",id)
        self.queue_manager.append_queue("queue_list",id)
        self.logger.info("SFM Job Published with ID {}".format(id))
        
    def publish_nerf_job(self, id: str, vid: Video, sfm: Sfm, config: TrainingConfig):
        """
        Publishes a new job to the nerf-in queue hosted on RabbitMQ.
        Image sets are converted to links to be downloaded by the nerf worker

        Args:
            id (str): job id
            vid (Video): video object
            sfm (Sfm): sfm object
            config (TrainingConfig): job specific training config object
        """
        
        job = {
            "id": id,
            "vid_width": vid.width if vid.width else 0,
            "vid_height": vid.height if vid.height else 0,
        }
        
        # replace relative filepaths with URLS
        sfm_data = sfm.to_dict()
        for i,frame in enumerate(sfm_data["frames"]):
            file_path = frame["file_path"]
            file_url = self.to_url(file_path)
            sfm_data["frames"][i]["file_path"] = file_url
        
        # Merge job specific, video. and sfm data into job details
        combined_job = {**job, **sfm_data, **config.nerf_config}
        json_job = json.dumps(combined_job)

        # Publish job to nerf-in queue and append to nerf_list queue manager
        self.channel.basic_publish(exchange='', routing_key='nerf-in', body=json_job)
        self.queue_manager.append_queue("nerf_list", id)
        self.logger.info("NERF Job Published with ID %s", id)

    def start_consumers(self):
        """
        Initiates the consumer threads for the sfm-out and nerf-out queues.
        """
        self.logger.info("RabbitMQServiceV2: Starting consumer threads for xxx-out queues")
        
        sfm_thread = threading.Thread(target=self._consume_sfm_out)
        nerf_thread = threading.Thread(target=self._consume_nerf_out)
        sfm_thread.start()
        nerf_thread.start()

    def _consume_sfm_out(self):
        """
        Internal function to consume from the sfm-out queue and process the finished jobs.
        Stores the video and sfm data in the database and publishes a new job to the nerf-in queue.
        
        TODO: Modify this and Colmap Worker code to better resemble nerf worker code
        """
        
        def process_sfm_job(ch, method, properties, body):
            """
            Handles the processing of a single sfm job from the sfm-out queue.

            Args:
                ch (_type_): piak channel object
                method (_type_): pika method object
                properties (_type_): message properties
                body (_type_): message body
            """
            #load queue object
            sfm_data = json.loads(body.decode())
            flag = sfm_data['flag']
            id = sfm_data['id']
            self.logger.info("SFM TASK RETURNED WITH FLAG {}".format(flag))
            # Process frames only if video is valid (== 0)
            if(flag == 0):
            #convert each url to filepath
            #store png 
                for i,fr_ in enumerate(sfm_data['frames']):
                    # TODO: This code trusts the file extensions from the worker
                    # TODO: handle files not found
                    url = fr_['file_path']
                    self.logger.log(logging.INFO, f"Downloading image from {url}")
                    img = requests.get(url)
                    url_path = urlparse(fr_['file_path']).path
                    filename = url_path.split("/")[-1]
                    file_path =  "data/sfm/" + id 
                    os.makedirs(file_path, exist_ok=True) 
                    file_path += "/" + filename
                    open(file_path,"wb").write(img.content)

                    path = os.path.join(os.getcwd(), file_path)
                    sfm_data['frames'][i]["file_path"] = file_path
            
            # Get indexes of k mean grouped frames
            #k_sampled = k_mean_sampling(sfm_data)

            # Use those frames to revise list of frames used in sfm generation
            #sfm_data['frames'] = [sfm_data['frames'][i] for i in k_sampled]

            del sfm_data['flag']
            #call SceneManager to store to database
            vid = Video.from_dict(sfm_data)
            sfm = Sfm.from_dict(sfm_data)
            self.scene_manager.set_sfm(id,sfm)
            self.scene_manager.set_video(id,vid)
            config = self.scene_manager.get_training_config(id)

            #remove video from sfm_list queue manager
            self.queue_manager.pop_queue("sfm_list",id)

            self.logger.info("Saved finished SFM job")
            new_data = json.dumps(sfm_data)
            
            # Publish new job to nerf-in only if good status (flag of 0)
            if(flag == 0):
                self.publish_nerf_job(id, vid, sfm, config)
            else:
                self.queue_manager.pop_queue("queue_list",id)
                # Set a specific flag to the failed flag (normal is 0)
                nerf = NerfV2().from_dict({"flag":flag})
                # Set this to the final output
                self.scene_manager.set_nerfV2(id, nerf)

            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        self._consume_queue('sfm-out', process_sfm_job)

    def _consume_nerf_out(self):
        """ 
        Internal function to consume from the nerf-out queue and process the finished jobs.
        Stores the nerf data in the database and sends the final response to the client.
        """
        
        def process_nerf_job(ch, method, properties, body):
            """
            Handles the processing of a single nerf job from the nerf-out queue.
            Args:
                ch (_type_): pika channel object
                method (_type_): pika method object
                properties (_type_): message properties
                body (_type_): message body
            """
            nerf_data = json.loads(body.decode())
            id = nerf_data["id"]
            
            nerf = self.scene_manager.get_nerfV2(id)
            if nerf:
                nerf = nerf.to_dict()
            else:
                self.logger.warning("Could not find nerf object for id %s, creating a new one", id)
                nerf = NerfV2().get_empty().to_dict()
                
            self.logger.info("pnj: Nerf is %s", nerf)
                
            # Extract output endpoints and config data
            output_endpoints = nerf_data["output_endpoints"]        
            config = self.scene_manager.get_training_config(id).nerf_config
            output_types = config["output_types"]
            save_iterations = config["save_iterations"]
            output_path = Path(f"data/nerf/{id}")
            
        
            # Save all generated output resources to local file storage
            for endpoint_type in output_endpoints.keys():
                
                self.logger.info("nerf: %s", nerf)                
                
                if f"{endpoint_type}_file_paths" not in nerf:
                    nerf[f"{endpoint_type}_file_paths"] = {}
                
                self.logger.info("Retrieving Nerf output type %s", endpoint_type)
                
                if endpoint_type not in output_types:
                    self.logger.warning(
                        "Mismatch in nerf output and config. "
                        "Saving unwanted resource of type %s. Wanted nerf"
                        "output types %s. Job %s.", 
                        endpoint_type, str(output_types), id)
                
                extension = ""
                if endpoint_type == "splat_cloud":
                    extension = "splat"
                elif endpoint_type == "point_cloud":
                    extension = "ply"
                elif endpoint_type == "video":
                    extension = "mp4"
                elif endpoint_type == "model":
                    extension = "th"
                else:
                    self.logger.critical("Unexpected endpoint type received. Skipping Saving. Job %s", id)
                    continue
                
                for iteration in output_endpoints[endpoint_type]["save_iterations"]:
                    
                    if iteration not in config["save_iterations"]:
                        self.logger.warning(
                            "Mismatch in nerf output and config."
                            "Saving unwanted iteration %s of resource type %s."
                            "Wanted iterations %s. Job %s",
                            iteration, endpoint_type, str(save_iterations.items()))
                    
                    endpoint = output_endpoints[endpoint_type]["endpoint"]
                    response = requests.get(endpoint, params={"iteration": iteration})
                    
                    if not response.status_code == 200:
                        message = f"Request to {endpoint} returned unsucessfully. Aborting saving this file. Job {id}"
                        try:
                            if "error" in (_json := json.loads(response.content)):
                                message += f"\nerror: {_json['error']}"
                        except: 
                            self.logger.critical("Failure to read response. Was not a file type, and could not be dumped to JSON")
                        self.logger.warning(message)
                        continue
                    
                    
                    # Save to data/nerf/{id}/{type}/iteration_{iteration}/{id}.{extension}
                    file_path = output_path / f"{endpoint_type}/iteration_{iteration}/{id}.{extension}"
                    os.makedirs(file_path.parent, exist_ok=True)
                    open(file_path, "wb").write(response.content)       
                    
                    # Write paths to Nerf db obj
                    nerf[f"{endpoint_type}_file_paths"][iteration] = file_path
                    
            # TODO: Probably should modify flag when saving
            
            nerf["flag"] = 0
            
            self.scene_manager.set_nerfV2(id, NerfV2().from_dict(nerf))
            
            #remove video from nerf_list and queue_list (end of full process)
            self.queue_manager.pop_queue("nerf_list",id)
            self.queue_manager.pop_queue("queue_list",id)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self._consume_queue('nerf-out', process_nerf_job)

    def _consume_queue(self, queue_name, callback):
        """
        Internal function that consumes from the specified queue and calls the callback function for each message.
        Creates a new connection every time this function is called.
        
        Args:
            queue_name (_type_): Name of the queue to consume from
            callback (function): Function to call for each message
        """
        self.logger.info("RabbitMQServiceV2: Starting consumer for queue: %s", queue_name)
        
        connection = pika.BlockingConnection(self.parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=queue_name, on_message_callback=callback)
        
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()
        finally:
            connection.close()



# TODO: make rabbitmq resistent to failed worker jobs
@deprecated("Legacy code. Use RabbitMQServiceV2 to publish and consume from RabbitMQ in a single class")
class RabbitMQService:
    # TODO: Communicate with rabbitmq server on port defined in web-server arguments
    """
    RabbitMQService is a class that communicates with the RabbitMQ server to publish new jobs to the workers.
    """
    def __init__(self, rabbitip, manager):
        """
        Initializes the RabbitMQService class by connecting to the RabbitMQ server and creating a channel to publish jobs to the workers.
        Composes QueueListManager to view and manage job queues.
        """
        self.logger = logging.getLogger('web-server')
        rabbitmq_domain = rabbitip
        credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
        parameters = pika.ConnectionParameters(rabbitmq_domain, 5672, '/', credentials, heartbeat=300)
        self.queue_manager = manager
        
        #2 minute timer
        timeout = time.time() + 60 * 2

        #retries connection until conencts or 2 minutes pass
        while True:
            if time.time() > timeout:
                self.logger.critical("RabbitMQService, _init_, took too long to connect to rabbitmq")
                raise Exception("RabbitMQService, _init_, took too long to connect to rabbitmq")
            try:
                self.connection = pika.BlockingConnection(parameters)  
                self.channel = self.connection.channel() 
                self.channel.queue_declare(queue='sfm-in')
                self.channel.queue_declare(queue='nerf-in')
                break
            except pika.exceptions.AMQPConnectionError:
                continue

        #TODO: make this dynamic from config file
        self.base_url = "http://localhost:5000/"
        # for docker
        self.base_url = "http://host.docker.internal:5000/"
        # for queue list positions

    def to_url(self,file_path):
        """
        Converts a file path to a URL for the worker to download the file from the web server.
        """
        return self.base_url+"/worker-data/"+file_path


    def publish_sfm_job(self, id: str, vid: Video ):
        """
        Publishes a new job to the sfm-in queue hosted on RabbitMQ.

        Args:
            id (str): job id
            vid (Video): video object
        """
        job = {
            "id": id,
            "file_path": self.to_url(vid.file_path)
        }
        json_job = json.dumps(job)
        self.channel.basic_publish(exchange='', routing_key='sfm-in', body=json_job)
        # add to sfm_list and queue_list (first received, goes into overarching queue) queue manager
        self.queue_manager.append_queue("sfm_list",id)
        self.queue_manager.append_queue("queue_list",id)
        
        self.logger.info("SFM Job Published with ID {}".format(id))
           
    def publish_nerf_job(self, id: str, vid: Video, sfm: Sfm):
        """
        Publishes a new job to the nerf-in queue hosted on RabbitMQ.
        Image sets are converted to links to be downloaded by the nerf worker

        Args:
            id (str): job id
            vid (Video): video object
            sfm (Sfm): sfm object
        """
        job = {
            "id": id,
            "vid_width": vid.width if vid.width else 0,
            "vid_height": vid.height if vid.height else 0,
        }

        # replace relative filepaths with URLS
        sfm_data = sfm.to_dict()
        for i,frame in enumerate(sfm_data["frames"]):
            file_path = frame["file_path"]
            file_url = self.to_url(file_path)
            sfm_data["frames"][i]["file_path"] = file_url
        
        combined_job = {**job, **sfm_data}
        json_job = json.dumps(combined_job)
        self.channel.basic_publish(exchange='', routing_key='nerf-in', body=json_job)
        # add to nerf_list queue manager
        self.queue_manager.append_queue("nerf_list",id)

        self.logger.info("NERF Job Published with ID {}".format(id))

    #call
    #each sfm_out object would be in the form
        # "id" = id
        # "vid_width": int vid.width,
        # "vid_height": int vid.height
        # "intrinsic_matrix": float[]
        # "frames" = array of urls and extrinsic_matrix[float]
    #   channel.basic.consume(on_message_callback = callback_sfm_job, queue = sfm_out)

# TODO: Should probably find a better place for this function. Should really be in sfm worker code
def find_elbow_point(data, max_k=35):
    """
    Finds the elbow point of a kmeans graph using the Within-Cluster Sum of Squares (WCSS) method.

    Args:
        data (_type_): graph data
        max_k (int, optional): Maximum number of clusters to consider. Defaults to 35.
    """
    # Within-Cluster Sum of Squares (WCSS)
    wcss = []

    # Set a maximum limit for computational efficiency
    max_k = min(len(data), max_k)  

    # Check if max_k is very large
    max_k = max(max_k, math.floor(math.sqrt(len(data))))

    # Calculate WCSS for different values of k
    for k in range(1, max_k + 1):
        kmeans = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Fill in x values for elbow function 
    x = range(1, len(wcss)+1)

    # Determine Elbow point of graph
    #TODO: fix this
    #elbow = kneed.KneeLocator(x, wcss, curve = 'convex', direction='decreasing')
    
    # Returns elbow point (along with x and y values for graph testing)
    #return elbow.knee, x, wcss

# TODO:  Should probably find a better place for this function. Should really be in sfm worker code
def k_mean_sampling(frames, size=100):
    """
    Perform k-means clustering on the spherical coordinates 
    of the extrinsic matrix of each frame in the video.

    Args:
        frames (_type_): Input data containing extrinsic matrices of each frame.
        size (int, optional): Number of frames to sample. Defaults to 100.
    Returns:
        _type_:  Selected frames from the input data. One frame per cluster.
    """
    logger = logging.getLogger('web-server')

    #TODO Make this input passed in, with default value 100
    CLUSTERS = size

    extrins = []
    angles = []
    for f in frames["frames"]:
        extrinsic = np.array(f["extrinsic_matrix"])
        extrins+=[ extrinsic ]
    for i,e in enumerate(extrins):

        # t == rectangular coordinates
        t = e[0:3,3]

        # s == spherical coordinates

        # r = sqrt(x^2 + y^2 + z^2)
        r = math.sqrt((t[0]*t[0])+(t[1]*t[1])+(t[2]*t[2]))
        theta = math.acos(t[2]/r)
        phi = math.atan(t[1]/t[0])

        #convert radian to degrees

        theta = (theta * 180) / math.pi
        phi = (phi * 180) / math.pi

        s = [theta,phi]

        angles.append(s)

    #elbow_point, _, _ = find_elbow_point(angles)
    elbow_point = 10
    km = sklearn.cluster.Kmeans(n_clusters=elbow_point, n_init=10)
    km.fit(angles)

    labels = km.labels
    if (len(set(labels)) != elbow_point):
        logger.error("Error with clustering.")

    cluster_array = [ [] for _ in range(elbow_point) ]

    for i in range(len(angles)):
        cluster_array[labels[i]].append(i)

    centroids = km.cluster_centers_
    closest_frames = []

    # Find the frame closest to each centroid in each cluster
    for idx, cluster_indices in enumerate(cluster_array):

        # Extract data points belonging to the current cluster
        cluster_data = np.array([angles[i] for i in cluster_indices])
        
        # Calculate the centroid of the current cluster
        centroid = centroids[idx]

        # Calculate the distances between each data point and the centroid
        distances = np.linalg.norm(cluster_data - centroid, axis=1)

        # Find the index of the closest frame within the current cluster
        closest_frame_index = cluster_indices[np.argmin(distances)]
        
        # Append the index of the closest frame to the list
        closest_frames.append(closest_frame_index)

    return closest_frames

# TODO: Turn this into its own class. Makes sense for if you had multiple web-servers being load balanced
@deprecated("Legacy code. Use RabbitMQServiceV2 to consume from RabbitMQ in a single class")
def digest_finished_sfms(rabbitip, rmqservice: RabbitMQService, scene_manager: SceneManager, queue_manager: QueueListManager):
    logger = logging.getLogger('web-server')

    def process_sfm_job(ch,method,properties,body):
        #load queue object
        sfm_data = json.loads(body.decode())
        flag = sfm_data['flag']
        id = sfm_data['id']
        logger.info("SFM TASK RETURNED WITH FLAG {}".format(flag))
        # Process frames only if video is valid (== 0)
        if(flag == 0):
        #convert each url to filepath
        #store png 
            for i,fr_ in enumerate(sfm_data['frames']):
                # TODO: This code trusts the file extensions from the worker
                # TODO: handle files not found
                url = fr_['file_path']
                logger.log(logging.INFO, f"Downloading image from {url}")
                img = requests.get(url)
                url_path = urlparse(fr_['file_path']).path
                filename = url_path.split("/")[-1]
                file_path =  "data/sfm/" + id 
                os.makedirs(file_path, exist_ok=True) 
                file_path += "/" + filename
                open(file_path,"wb").write(img.content)

                path = os.path.join(os.getcwd(), file_path)
                sfm_data['frames'][i]["file_path"] = file_path
        
        # Get indexes of k mean grouped frames
        #k_sampled = k_mean_sampling(sfm_data)

        # Use those frames to revise list of frames used in sfm generation
        #sfm_data['frames'] = [sfm_data['frames'][i] for i in k_sampled]

        del sfm_data['flag']
        #call SceneManager to store to database
        vid = Video.from_dict(sfm_data)
        sfm = Sfm.from_dict(sfm_data)
        scene_manager.set_sfm(id,sfm)
        scene_manager.set_video(id,vid)

        #remove video from sfm_list queue manager
        queue_manager.pop_queue("sfm_list",id)

        logger.info("Saved finished SFM job")
        new_data = json.dumps(sfm_data)
        
        # Publish new job to nerf-in only if good status (flag of 0)
        if(flag == 0):
            rmqservice.publish_nerf_job(id, vid, sfm)
        else:
            queue_manager.pop_queue("queue_list",id)
            # Set a specific flag to the failed flag (normal is 0)
            nerf = Nerf().from_dict({"flag":flag})
            # Set this to the final output
            scene_manager.set_nerf(id, nerf)


        ch.basic_ack(delivery_tag=method.delivery_tag)
        

    # create unique connection to rabbitmq since pika is NOT thread safe
    rabbitmq_domain = rabbitip
    credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(rabbitmq_domain, 5672, '/', credentials, heartbeat=300)

    #2 minute timer
    timeout = time.time() + 60 * 2

    #retries connection until connects or 2 minutes pass
    while True:
        if time.time() > timeout:
            logger.critical("digest_finished_sfms took too long to connect to rabbitmq")
            raise Exception("digest_finished_sfms took too long to connect to rabbitmq")
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue='sfm-out')

            # Will block and call process_sfm_job repeatedly
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='sfm-out', on_message_callback=process_sfm_job)
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
                break
        except pika.exceptions.AMQPConnectionError:
            continue

# TODO: Turn this into its own class. Makes sense for if you had multiple web-servers being load balanced
@deprecated("Legacy code. Consume tensoRF output. Use RabbitMQServiceV2 to consume from RabbitMQ in a single class")
def digest_finished_nerfs(rabbitip, rmqservice: RabbitMQService, scene_manager: SceneManager, queue_manager: QueueListManager):
    logger = logging.getLogger('web-server')

    def process_nerf_job(ch,method,properties,body):
        
        nerf_data = json.loads(body.decode())
        video = requests.get(nerf_data['rendered_video_path'])
        id = nerf_data['id']
        
        filepath = "data/nerf/" 
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath+f"{id}.mp4")
        
        open(filepath,"wb").write(video.content)

        nerf_data["flag"] = 0
        nerf_data['rendered_video_path'] = filepath
        id = nerf_data['id']
        
        # Static method to create Nerf object from dictionary
        nerf = Nerf().from_dict(nerf_data)
        scene_manager.set_nerf(id, nerf)

        #remove video from nerf_list and queue_list (end of full process) queue manager
        queue_manager.pop_queue("nerf_list",id)
        queue_manager.pop_queue("queue_list",id)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    # create unique connection to rabbitmq since pika is NOT thread safe
    rabbitmq_domain = rabbitip
    credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(rabbitmq_domain, 5672, '/', credentials,heartbeat=300)

    #2 minute timer
    timeout = time.time() + 60 * 2

    #retries connection until connects or 2 minutes pass
    while True:
        if time.time() > timeout:
            logger.critical("digest_finished_nerfs took too long to connect to rabbitmq")
            raise Exception("digest_finished_nerfs took too long to connect to rabbitmq")
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue='nerf-out')

            # Will block and call process_nerf_job repeatedly
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue='nerf-out', on_message_callback=process_nerf_job)
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
                break
        except pika.exceptions.AMQPConnectionError:
            continue
        
