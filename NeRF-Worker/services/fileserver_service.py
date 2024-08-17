"""
This file contains the FileServer class which is responsible for handling all GET requests to the nerf-worker server.

TODO: Setup CORS, so only webserver can use these endpoints
"""


import logging

from log import nerf_worker_logger

from typing import Optional
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, Response, request, make_response

cert_path = '/app/secrets/cert.pem' # Local Self-Signed Cert Path

class FileServer:
    """
    Main webserver for the nerf gaussian worker. Handles all GET endpoints
    """

    def __init__(self, base_url: str = "http://nerf-worker-gaussian:5200/"):
        self.app = Flask(__name__)
        self.app.url_map.strict_slashes = False
        self.base_url = base_url
        self.logger = nerf_worker_logger("nerf-worker-server")
        self.setup_routes()
        self.logger.info("FileServer initialized")

    def setup_routes(self):
        """
        Creates flask routes for GET endpoints
        """
        self.app.route("/data/nerf/video/<uuid>", methods=["GET"])(
            self.send_video)
        
        self.app.route("/data/nerf/point_cloud/<uuid>", methods=["GET"])(
            self.send_point_cloud)
        
        self.app.route("/data/nerf/splat_cloud/<uuid>", methods=["GET"])(
            self.send_splat_cloud)
        
        self.app.route("/data/nerf/<file_type>/<uuid>", methods=["GET"])(
            self.send_output)
        
        self.logger.info("Routes set up")

    def get_latest_iteration(self, uuid: str, file_type: str, file_extension: str) -> Optional[Path]:
        """
        Finds the farthest trained iteration path for give output type and extension

        TODO: DONE 7/10/24 Change from point_cloud to both ply and splat, as there might 
        be only a iteration_X/point_cloud/uuid.ply but not a same/path/uuid.splat

        Args:
            uuid (_type_): job ID
            file_type (_type_): Overarching information representation
            method of the file. "point_cloud" or "video" 
            file_extension (str): Specific file extension to look for (e.g., ".ply", ".splat", ".mp4").
        Returns:
            Optional[Path]: The furthest training iteration for which the file type exists
            Returns the relative path to the associated file.
        """
        self.logger.info("Getting latest iteration for UUID: %s, file type: %s, file extension: %s",
                         uuid, file_type, file_extension)

        base_dir = Path("data/nerf") / uuid / file_type

        if not base_dir.exists():
            self.logger.warning("Base directory not found: %s", base_dir)
            return None

        # Get all iteration directories in rerverse by iteration
        iteration_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir()
             and d.name.startswith("iteration_")],
            key=lambda x: int(x.name.split("_")[1]),
            reverse=True
        )

        # Find latest valid (existing) path
        for iteration_dir in iteration_dirs:
            file_path = iteration_dir / f"{uuid}.{file_extension}"
            if file_path.exists():
                self.logger.info(
                    "Latest iteration found: %s with file: %s", iteration_dir.name, file_path.name)
                return iteration_dir

        self.logger.warning("No iterations found with required file for UUID: %s, file type: %s, file extension: %s",
                            uuid, file_type, file_extension)
        return None

    def send_file(self, uuid: str, file_type: str, file_extension: str, iteration: int = None) -> Response:
        """ 
        Sends output files as specified by GET endpoints.
        Performs path translation from GET endpoints to the actual
        storage location of the requested resource.

        Args:
            uuid (str): job id
            storage_type(str): The overaching information representation 
            of the file, e.g. both .ply and .splat files are point_clouds.
            file_extension (str): Specific file type
            iteration (int, optional): The iteration of training for which
            to retrieve the resource. If None, then uses farthest iteration.
            Defaults to None. 

        Returns:
            Response: Response object containing either file contents or error
        """
        self.logger.info("Sending file for UUID: %s, file type: %s, extension: %s, iteration: %s",
                         uuid, file_type, file_extension, iteration)

        if iteration:
            file_path = Path("data/nerf") / uuid / file_type / \
                f"iteration_{iteration}" / f"{uuid}.{file_extension}"
        else:
            latest_iter = self.get_latest_iteration(
                uuid, file_type, file_extension)
            if not latest_iter:
                self.logger.error("No latest iteration resource found for UUID: %s, file type: %s, extension: %s",
                                  uuid, file_type, file_extension)
                return make_response(jsonify({
                    "error": f"No latest iteration resource of local path data/nerf/{uuid}/{file_type}/{file_extension} found"
                }), 400)

            file_path = latest_iter / f"{uuid}.{file_extension}"
        file_path = file_path.absolute()

        if not file_path.exists():
            self.logger.error("File not found: %s", file_path)
            return make_response(jsonify({
                "error": f"No iteration {iteration} resource of local path data/nerf/{uuid}/{file_type}/{file_extension} found"
            }), 404)

        self.logger.info("Sending file: %s", file_path)
        return send_from_directory(file_path.parent, file_path.name)

    def send_video(self, uuid: str) -> Response:
        """
        Handles incoming get requests for trained nerf data in form of video

        Args:
            uuid (str): Requested UUID

        Returns:
            : Video data from the requested URI
        """
        iteration = request.args.get('iteration')
        self.logger.info(
            "Received request for video, UUID: %s, Iteration: %s", uuid, iteration)
        return self.send_file(uuid, "video", "mp4", iteration)

    def send_point_cloud(self, uuid: str):
        """
        Handles incoming get requests for trained nerf data in form of point cloud

        Args:
            uuid (_type_): Requested UUID

        Returns:
            _type_: Point cloud data from the requested URI
        """
        iteration = request.args.get('iteration')
        self.logger.info(
            "Received request for point cloud, UUID: %s,Iteration: %s", uuid, iteration)
        return self.send_file(uuid, "point_cloud", "ply", iteration)

    def send_splat_cloud(self, uuid: str) -> Response:
        """
        Handles incoming get requests for trained nerf data that
        has been converted to .splat notation for front end rendering

        Args:
            uuid (_type_): Requested UUID

        Returns:
            _type_: Splat data from the requested URI
        """
        iteration = request.args.get('iteration')
        self.logger.info(
            "Received request for splat cloud, UUID: %s, Iteration: %s", uuid, iteration)
        return self.send_file(uuid, "point_cloud", "splat", iteration)

    def send_output(self, file_type: str, uuid: str) -> Response:
        """
        Handles incoming get requests for trained nerf data
        in various output forms

        Args:
            file_type (str): Type of file requested
            uuid (str): Request UUID

        Returns:
            _type_: Data for the requested resource
        """

        iteration = request.args.get('iteration')
        self.logger.info(
            "Received request for output, file type: %s, UUID: %s, iteration: %s", file_type, uuid, iteration)
        if file_type == "video":
            return self.send_file(uuid, "video", "mp4", iteration)
        if file_type == "point_cloud":
            return self.send_file(uuid, "point_cloud", "ply", iteration)
        if file_type == "splat":
            return self.send_file(uuid, "point_cloud", "splat", iteration)

        else:
            self.logger.error("Invalid file type requested: %s", file_type)
            return jsonify({"error": "Invalid file type"}), 400

    def start(self, host="0.0.0.0", port=5200, debug=False) -> None:
        """
        Start the Flask server, open for business

        Args:
            host (str, optional: ipv4 to host device. Defaults to "0.0.0.0".
            port (int, optional): port. Defaults to 5200.
            debug (bool, optional): Allows hot reloading. Defaults to False.
        """
        self.logger.info("Starting FileServer on %s:%s", host, port)
        self.app.run(host=host, port=port, debug=debug)


def start_flask():
    """
    Starts the flask server for the nerf-worker to communicate with the web-server
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("nerf-worker-gaussian")
    logger.info("Starting Flask server")
    server = FileServer()
    server.start()
