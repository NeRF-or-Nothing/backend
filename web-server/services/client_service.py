"""
This module contains the ClientService class, which is responsible for handling all incoming client requests to the web-server.

TODO: DONE 8/7/24 Why call this ClientService but file scene_service.py?
TODO: Test all Scene.py, scene_Service.py, controller.py 
TODO: Unify Error Handling across all services and workers
"""


import os
import logging

from models.scene import Video, TrainingConfig
from models.managers import SceneManager, UserManager
from models.status import NerfError, NerfStatus, UserError, UserStatus
from services.queue_service import RabbitMQService, RabbitMQServiceV2
from utils.response_utils import create_response

from functools import wraps
from typing_extensions import deprecated
from typing import Optional, Tuple
from uuid import uuid4
from werkzeug.utils import secure_filename
from flask import Response, send_file


class ClientService:
    def __init__(self, scene_manager: SceneManager, rmqservice: RabbitMQServiceV2, user_manager: UserManager):
        self.logger = logging.getLogger('web-server')
        self.scene_manager = scene_manager
        self.rmqservice = rmqservice
        self.user_manager = user_manager

    def verify_user_access(f):
        """
        Decorator to verify user access to a specific job/resource.
        Will insert argument job_id at the beginning of the function call.
        """
        @wraps(f)
        def decorated(self, user_id: str, job_id: str, *args, **kwargs):
            if not self.user_manager.user_has_job_access(user_id, job_id):
                return create_response(
                    status=NerfStatus.ERROR,
                    error=UserError.UNAUTHORIZED,
                    message="User does not have access to this resource",
                    status_code=403
                )
            return f(self, job_id, *args, **kwargs)
        return decorated

    @verify_user_access
    def get_nerf_metadata(self, uuid: str) -> Response:
        """
        Retrieve all metadata for the given uuid (job id).
        Return formatted JSON response. Sends file
        size, chunk size, and number of chunks for each
        resource type. Will respond 404 until job is finished.
        
        TODO Change this to return training progress if not finished, and nerf metadata if finished \
            (or just direct frotend to use /queue endpoints until job is finished)
        
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

    @verify_user_access
    def get_nerf_type_metadata(self, uuid: str, output_type: str) -> Response:
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

    def handle_incoming_video(self, user_id, video_file, request_params: dict, scene_name: Optional[str]) -> Optional[str]:
        """
        Validates POSTed videos, finishes job creation, and posts it to worker
        pipeline. Assumes valid user_id given.

        TODO: Add .wav file support, we claim to support .mp4 and .wav files

        Args:
            user_id (str): User ID
            video_file (_type_): video file
            request_params (dict): additional job parameters from 
            incoming POST request
            scene_name (str): Optional scene name
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

        uuid = str(uuid4())
        self.logger.info("New UUID : %s", uuid)

        # Save video to file storage
        video_name = uuid + ".mp4"
        videos_folder = "data/raw/videos"
        video_file_path = videos_folder
        os.makedirs(videos_folder, exist_ok=True)
        video_file_path = os.path.join(video_file_path, video_name)
        video_file.save(video_file_path)

        # Save video to database and create config
        video = Video(video_file_path)
        training_config = TrainingConfig().get_default()
        training_config.update({"nerf_config": request_params})
        
        self.scene_manager.set_video(uuid, video)
        self.scene_manager.set_scene_name(uuid, scene_name)
        self.scene_manager.set_training_config(uuid, training_config)
        self.rmqservice.publish_sfm_job(uuid, video, training_config)
        
        user = self.user_manager.get_user_by_id(user_id)
        user.add_scene(uuid)
        self.user_manager.update_user(user)
        
        return uuid

    @verify_user_access
    def send_nerf_resource(self, uuid: str, resource_type: str, iteration: str, range_header: Optional[str]) -> Response:
        """
        Handles actual file retrieval and file sending for incoming 
        GET requests to WebServer routes for finished nerf training
        resources. You SHOULD request a range header to get the file,
        as max http is 30MB, and if not provided, sends the first
        30MB or less of the file.

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
    
    def get_user_history(self, user_id: str) -> Response:
        """
        List all finished training resource uuids for a specific user. 
        Verification occurs in webserver.

        Args:
            user_id (str): The ID of the user.
        Returns:
            A response containing the list of resource UUIDs.
        """
        all_resources = []
        
        user = self.user_manager.get_user_by_id(user_id) 
        
        if user == UserError.USER_NOT_FOUND:
            return create_response(
                status=UserStatus.ERROR,
                error=UserError.USER_NOT_FOUND,
                message="User not found",
                status_code=404
            )
        
        for scene_id in user.scene_ids:
            nerf = self.scene_manager.get_nerfV2(scene_id)
            if nerf:
                all_resources.append(scene_id)
        
        return create_response(
            status=UserStatus.SUCCESS,
            error=UserError.NO_ERROR,
            message="User found, resources retrieved",
            data={"resources": all_resources},
            status_code=200
        )
        
    @verify_user_access
    def get_preview(self, uuid: str):
        """
        Returns the preview image and scene name for the given job UUID.
        
        Args:
            uuid (str): The job UUID.
        Returns:
            Response: response containing the preview image and scene name.
        """
        
        sfm = self.scene_manager.get_sfm(uuid)
        if sfm:
            sfm_folder = f"data/sfm/{uuid}"
            png_files = [f for f in os.listdir(sfm_folder) if f.endswith(".png")]
            if png_files:
                scene_name = self.scene_manager.get_scene_name(uuid)
                file_path = os.path.join(sfm_folder, png_files[0])
                
                response =  send_file(
                    os.path.abspath(file_path),
                    mimetype='image/png',
                    as_attachment=True,
                    download_name=png_files[0],
                )
                response.headers['X-Scene-Name'] = scene_name
                response.headers['Access-Control-Expose-Headers'] = 'X-Scene-Name'
                return response
            
        return create_response(
            status=NerfStatus.ERROR,
            error=NerfError.FILE_NOT_FOUND,
            message="No preview image found",
            uuid=uuid,
            status_code = 404
        )
        
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
        self.logger.debug("nerf: %s", nerf)

        # Find path for given iteration, or latest available
        if iteration == "" or iteration is None:
            self.logger.info("No iteration given, sending latest")
            valid_iterations = nerf[f"{type}_file_paths"]
            max_iteration = max(valid_iterations.keys(), key=lambda x: int(x))
            path = valid_iterations[max_iteration]
            return path
        if int(iteration) in nerf_config["save_iterations"]:
            self.logger.info("Valid iteration given, sending resource")
            path = nerf[f"{type}_file_paths"][iteration]
            return path
        else:
            return None

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
    