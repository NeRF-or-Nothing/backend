"""
This module contains the ClientService class, which is responsible for handling all incoming client requests to the web-server.

TODO: Why call this ClientService but file scene_service.py?
TODO: Test all Scene.py, scene_Service.py, controller.py 
TODO: Unify Error Handling across all services and workers
"""


import base64
import gzip
import json
import os
import io
import logging

from models.scene import Video, TrainingConfig
from models.managers import SceneManager, UserManager
from models.status import NerfError, NerfStatus
from services.queue_service import RabbitMQService, RabbitMQServiceV2
from utils.response_utils import create_response

from typing_extensions import deprecated
from typing import Optional, Tuple
from uuid import uuid4, UUID
from werkzeug.utils import secure_filename
from flask import jsonify, Response, send_file, make_response


class ClientService:
    def __init__(self, scene_manager: SceneManager, rmqservice: RabbitMQServiceV2):
        self.logger = logging.getLogger('web-server')
        self.scene_manager = scene_manager
        self.rmqservice = rmqservice

    def get_nerf_metadata(self, uuid: str) -> Response:
        """
        Retrieve all metadata for the given UUID.
        Return formatted JSON response. Sends file
        size, chunk size, and number of chunks for each
        resource type.
        
        Args:
            uuid (str): The UUID of the NeRF job.
        
        Returns:
            Response: A Flask response containing the metadata.
        """
        self.logger.info(f"Retrieving metadata for UUID {uuid}")
        try:
            nerf = self.scene_manager.get_nerfV2(uuid)
            if not nerf:
                self.logger.warning(f"No NeRF job found for UUID: {uuid}")
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.FILE_NOT_FOUND,
                    message=f"No NeRF job found for UUID: {uuid}",
                    uuid=uuid,
                    status_code=404
                )

            config = self.scene_manager.get_training_config(uuid)
            if not config:
                self.logger.warning(f"No configuration found for UUID: {uuid}")
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.INTERNAL_SERVER_ERROR,
                    message=f"No configuration found for UUID: {uuid}",
                    uuid=uuid,
                    status_code=500
                )

            self.logger.info(f"NeRF object for UUID {uuid}: {nerf.to_dict()}")
            self.logger.info(f"Config for UUID {uuid}: {config.to_dict()}")

            resources = {}
            for output_type in config.nerf_config['output_types']:
                resources[output_type] = {}
                file_paths = getattr(nerf, f"{output_type}_file_paths", {})
                for iteration, path in file_paths.items():
                    if os.path.exists(path):
                        file_size = os.path.getsize(path)
                        chunks = (file_size + (1024 * 1024 - 1)) // (1024 * 1024)  # Round up division
                        last_chunk_size = file_size % (1024 * 1024) or 1024 * 1024
                        resources[output_type][iteration] = {
                            "exists": True,
                            "size": file_size,
                            "chunks": chunks,
                            "last_chunk_size": last_chunk_size
                        }
                    else:
                        resources[output_type][iteration] = {"exists": False}

            return create_response(
                status=NerfStatus.READY,
                error=NerfError.NO_ERROR,
                message="Metadata retrieved successfully",
                uuid=uuid,
                data={"resources": resources},
                status_code=200
            )

        except Exception as e:
            self.logger.error(f"Error retrieving metadata for UUID {uuid}: {str(e)}")
            return create_response(
                status=NerfStatus.ERROR,
                error=NerfError.INTERNAL_SERVER_ERROR,
                message=f"Error retrieving metadata: {str(e)}",
                uuid=uuid,
                status_code=500
            )

    def get_nerf_type_metadata(self, output_type: str, uuid: str) -> Response:
        """
        Retrieve metadata for the specific output type and UUID.
        Return formatted JSON response.
        
        Args:
            output_type (str): The type of output (e.g., 'splat_cloud', 'point_cloud', 'video', 'model').
            uuid (str): The UUID of the NeRF job.
        
        Returns:
            Response: A Flask response containing the metadata for the specific output type.
        """
        self.logger.info("New request to retrieve metadata for UUID %s and output type %s", uuid, output_type)
        
        try:
            nerf = self.scene_manager.get_nerfV2(uuid)
            if not nerf:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.FILE_NOT_FOUND,
                    message=f"No NeRF job found for UUID: {uuid}",
                    uuid=uuid,
                    status_code=404
                )

            config = self.scene_manager.get_training_config(uuid)
            if not config or output_type not in config.nerf_config['output_types']:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.INVALID_INPUT,
                    message=f"Invalid output type: {output_type} for UUID: {uuid}",
                    uuid=uuid,
                    status_code=400
                )

            resources = {output_type:{}}
            file_paths = getattr(nerf, f"{output_type}_file_paths", {})
            for iteration, path in file_paths.items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path)
                    chunks = (file_size + (1024 * 1024 - 1)) // (1024 * 1024)  # Round up division
                    last_chunk_size = file_size % (1024 * 1024) or 1024 * 1024
                    resources[output_type][iteration] = {
                        "exists": True,
                        "size": file_size,
                        "chunks": chunks,
                        "last_chunk_size": last_chunk_size
                    }
                else:
                    resources[output_type][iteration] = {"exists": False}

            return create_response(
                status=NerfStatus.READY,
                error=NerfError.NO_ERROR,
                message=f"Metadata for {output_type} retrieved successfully",
                uuid=uuid,
                data={"resources": resources},
                status_code=200
            )

        except Exception as e:
            self.logger.error(f"Error retrieving metadata for UUID {uuid} and output type {output_type}: {str(e)}")
            return create_response(
                status=NerfStatus.ERROR,
                error=NerfError.INTERNAL_SERVER_ERROR,
                message=f"Error retrieving metadata: {str(e)}",
                uuid=uuid,
                status_code=500
            )

    def handle_incoming_video(self, video_file, request_params: dict) -> Optional[str]:
        """
        Validates POSTed videos, finishes job creation, and posts it to worker
        pipeline

        TODO: Add .wav file support, we claim to support .mp4 and .wav files

        Args:
            video_file (_type_): video file
            training_mode (str)): job nerf training mode
            output_types (List[str]]):  job output types
            **args (dict): additional job parameters from 
            incoming POST request

        Returns:
            str: UUID of new job
        """
        # Validate video file
        file_name = secure_filename(video_file.filename)
        if file_name == '':
            self.logger.error("ERROR: file not received")
            return None

        self.logger.info("Received new video input")

        file_ext = os.path.splitext(file_name)[1]
        if file_ext != ".mp4":
            self.logger.error("ERROR: improper file extension uploaded")
            return None

        # generate new id and save to file with db record
        uuid = str(uuid4())
        self.logger.info("New UUID : %s", uuid)

        # Save video to file storage
        video_name = uuid + ".mp4"
        videos_folder = "data/raw/videos"
        video_file_path = videos_folder
        os.makedirs(videos_folder, exist_ok=True)
        video_file_path = os.path.join(video_file_path, video_name)
        video_file.save(video_file_path)

        # Save video to database
        video = Video(video_file_path)
        self.scene_manager.set_video(uuid, video)

        # Save config to scenes collection. Used by workers for dynamic configs
        self.logger.debug("request_params: %s", request_params)
        training_config = TrainingConfig().get_default()
        training_config.update({"nerf_config": request_params})
        self.scene_manager.set_training_config(uuid, training_config)
        self.logger.info("Wrote config to mongodb. Config %s.\nJob %s", training_config.nerf_config, id)

        # create rabbitmq job for sfm
        self.rmqservice.publish_sfm_job(uuid, video, training_config)
        return uuid

    def send_nerf_resource(self, uuid: str, resource_type: str, iteration: str, range_header: Optional[str]) -> Response:
        """
        Handles actual file retrieval and file sending for incoming 
        GET requests to WebServer routes for finished nerf training
        resources.

        Args:
            id (str): Job UUID
            resource_type (str): Identifier string for resource type.
            "splat_cloud" | "point_cloud" | "model" | "video"
            iteration (str): Training progress snapshot iteration

        Returns:
            Response: Contains JSON with meta information and file data (if successful)
        """

        self.logger.info(f"Received request for resource {resource_type} for job {uuid}")

        resource_path = self.get_nerf_resource_path(uuid, resource_type, iteration)
        if not resource_path or not os.path.exists(resource_path):
            return create_response(
                status=NerfStatus.ERROR,
                error=NerfError.FILE_NOT_FOUND,
                uuid=uuid,
                status_code=404
            )

        file_size = os.path.getsize(resource_path)

        if range_header:
            try:
                start, end = self.parse_range_header(range_header, file_size)
            except ValueError as e:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.INVALID_RANGE,
                    message=str(e),
                    uuid=uuid,
                    status_code=416  # Range Not Satisfiable
                )
        else:
            start = 0
            end = file_size - 1

        chunk_size = end - start + 1

        with open(resource_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(chunk_size)

        response = Response(
            chunk,
            206 if range_header else 200,
            mimetype="application/octet-stream",
            direct_passthrough=True,
        )

        response.headers.set('Accept-Ranges', 'bytes')
        response.headers.set('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.set('Content-Length', str(chunk_size))

        return response

    def parse_range_header(self, range_header: str, file_size: int) -> Tuple[int, int]:
        """
        Parse the range header and return start and end bytes.
        
        Args:
            range_header (str): The Range header from the HTTP request.
            file_size (int): The total size of the file.
        
        Returns:
            Tuple[int, int]: The start and end byte positions.
        """
        try:
            h = range_header.replace('bytes=', '').split('-')
            start = int(h[0]) if h[0] else 0
            end = int(h[1]) if h[1] else file_size - 1
        except (IndexError, ValueError):
            raise ValueError("Invalid range header")
        
        if start >= file_size:
            raise ValueError("Range start exceeds file size")
        if end >= file_size:
            end = file_size - 1
        if start > end:
            raise ValueError("Invalid range: start > end")
        return start, end
    
    @deprecated("Legacy Code. Used for retriving old tensorf rendered videos")
    def get_nerf_video_path(self, uuid):
        """
        LEGACY. Finds file path to rendered video 

        Args:
            uuid (_type_): job id

        Returns:
            _type_: file path 
        """
        # TODO: depend on mongodb to load file path
        # return None if not found
        nerf = self.scene_manager.get_nerf(uuid)
        if nerf:
            return nerf.rendered_video_path
            # return ("Video ready", nerf.rendered_video_path)
        return None

    def list_resources_by_job_id(self, job_id: str) -> dict:
        """
        Lists all finished training resources for a specific job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            dict: A dictionary containing the job's finished resources.
        """
        raise NotImplementedError("Needs User, UserManager, and Job classes to be implemented more completely and implemented.")
        nerf = self.scene_manager.get_nerfV2(job_id)
        if nerf and nerf.flag == 0:  # Assuming flag 0 means processing is complete
            resources = {
                "splat_cloud": list(nerf.splat_cloud_file_paths.keys()) if nerf.splat_cloud_file_paths else [],
                "point_cloud": list(nerf.point_cloud_file_paths.keys()) if nerf.point_cloud_file_paths else [],
                "video": list(nerf.video_file_paths.keys()) if nerf.video_file_paths else [],
                "model": list(nerf.model_file_paths.keys()) if nerf.model_file_paths else []
            }
            return {job_id: resources}
        return {}
    
    
    def list_resources_by_user_id(self, user_id: str) -> dict:
        """
        NOT IMPLEMENTED. Would list all finished training resources for a specific user. 

        Args:
            user_id (str): The ID or API key of the user.

        Returns:
            dict: A dictionary containing all of the user's jobs and their finished resources.
        """
        raise NotImplementedError("Needs User, UserManager, and Job classes to be implemented more completely and implemented.")
        all_resources = {}
        
        # Assuming you have a method to get all job IDs for a user
        user = self.user_maneager.get_user(user_id) 
        job_ids = user.get_job_ids() # Prob use scenes list from user
        
        for job_id in job_ids:
            job_resources = self.get_resources_by_job_id(job_id)
            if job_resources:
                all_resources.update(job_resources)
        
        return all_resources


    def get_nerf_resource_path(self, uuid: str, type: str, iteration: str):
        """
        Retrieves the file path on web-server for valid training output
        type files. Outputs null if nerf object for uuid or requested resource 
        iteration does not exist. If no iteration provided, will retrieve the furthest
        trained resource of type.

        Args:
            uuid (str): Job id
            type (str): "point_cloud" | "splat_cloud" | "model" | "video"
            iteration (str): training iteration
        Returns:
            Optional[str]: local file path or None
        """
        self.logger.info("Retrieving resource path for job %s, type %s, iteration %s", uuid, type, iteration)
        
        nerf = self.scene_manager.get_nerfV2(uuid)
        nerf_config = self.scene_manager.get_training_config(uuid).nerf_config
        if not (nerf or nerf_config):
            self.logger.info("No nerf/nerf_config object found for job %s", uuid)
            return None
        nerf = nerf.to_dict()
        self.logger.info("nerf: %s", nerf)

        # Find path for given iteration, or latest available
        if iteration == "" or iteration is None:
            # No iteration given, send farthest trained resource
            self.logger.info("No iteration given, sending latest")
            valid_iterations = nerf[f"{type}_file_paths"]
            max_iteration = max(valid_iterations.keys(), key=lambda x: int(x))
            path = valid_iterations[max_iteration]
            return path
        if int(iteration) in nerf_config["save_iterations"]:
            # Valid Request parameter, attempt to find corresponding resource
            self.logger.info("Valid iteration given, sending resource")
            path = nerf[f"{type}_file_paths"][iteration]
            return path
        else:
            # Invalid Request parameter, return None
            return None

    @deprecated("Legacy Code. Used for retriving old tensorf rendered videos error flag")
    def get_nerf_flag(self, uuid):
        """
        Returns an integer describing the status of the video in the database.
        encode information on the COLMAP error that went wrong(e.g. 4 is a blurry video)
        """
        nerf = self.scene_manager.get_nerf(uuid)
        if nerf:
            return nerf.flag
        return 0
    
    def get_nerf_flagV2(self, uuid: str) -> int:
        """
        Returns the error flag for the given job UUID.

        Args:
            uuid (str): The job UUID.

        Returns:
            int: The error flag for the job.
        """
        nerf = self.scene_manager.get_nerfV2(uuid)
        if nerf:
            return nerf.flag
        return 0

    def decode_error_flag(self, flag: int) -> NerfError:
        """
        Decode the error flag into a NerfError enum.

        Args:
            flag (int): The error flag to decode.

        Returns:
            NerfError: The corresponding NerfError enum.
        """
        return next((error for error in NerfError if error.code == flag), NerfError.UNKNOWN)
