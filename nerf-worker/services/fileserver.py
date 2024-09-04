
"""
This file contains the FileServer class which is responsible for handling all GET requests to the nerf-worker server.

TODO: Setup CORS, so only webserver can use these endpoints
"""


import logging
import os

from log import nerf_worker_logger

from flask import Flask, send_from_directory, Response

class FileServer:
    """
    Main webserver for the nerf gaussian worker. Handles all GET endpoints.
    
    Note: If the request file root dir is in a parent directory of base_dir, the server will return a 404 error,
    even if os.path.exists() returns True.
    """

    def __init__(self, base_url: str = "http://nerf-worker-gaussian:5200/", base_dir: str = "/app"):
        self.app = Flask(__name__) 
        self.app.url_map.strict_slashes = False
        self.base_url = base_url
        self.base_dir = base_dir
        self.logger = nerf_worker_logger("nerf-worker-server")
        self.setup_routes()
        self.logger.info("FileServer initialized")

    def setup_routes(self):
        """
        Creates flask routes for GET endpoints
        """
        self.app.route("/data/nerf/<path:path>", methods=["GET"])(
            self.send_output)
        self.logger.info("Routes set up")

    def send_output(self, path) -> Response:
        """
        Handles GET requests for output files

        Args:
            path
        Returns:
            flask.Response: Data for the requested resource
        """
        self.logger.info("Received request for output: %s", path)
        full_path = os.path.join(self.base_dir, "data", "nerf", path)
        
        if os.path.exists(full_path):
            self.logger.debug("File exists: %s", full_path)
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
        else:
            self.logger.debug("File does not exist: %s", full_path)
            return "File not found", 404

    def start(self, host="0.0.0.0", port=5200, debug=False) -> None:
        """
        Start the Flask server, open for business.

        Args:
            host (str, optional): ipv4 to host device. Defaults to "0.0.0.0".
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
