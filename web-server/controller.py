"""
This module contains the WebServer class, which is responsible for handling incoming requests to the Flask server portion of web-server.
It is the main controller for the web-server, delegating tasks to the ClientService and QueueListManager classes.
Additionally handling worker data requests, user login and registration, and queue position requests.

TODO: Create authorization scheme for api keys. potential use custom headers in requests https://requests.readthedocs.io/en/latest/user/quickstart/#custom-headers 
TODO: Unify error handling and response messages
TODO: Create endpoint to list all available resources for a job id. Should prob follow call stack of get user data -> get all user jobs -> get resource data for job
TODO: Probably could add api key authentication for those resources, bc it should be quite easy to get user by api_key
TODO: Look into bcrypt for handling, dont have to store salt then
"""


import logging
import argparse
import os

from models.managers import UserManager, QueueListManager
from models.status import NerfError, NerfStatus
from utils.response_utils import create_response

from services.scene_service import ClientService
from pickle import TRUE
from typing import Optional, Tuple
from typing_extensions import deprecated
from uuid import UUID
from flask import Flask, request, make_response, send_file, send_from_directory, jsonify, Response
from flask_cors import CORS

def is_valid_uuid(value):
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


class WebServer:
    """
    The WebServer class is responsible for handling incoming requests to the Flask server portion of web-server.
    Delegates tasks to the ClientService and QueueListManager classes.
    """

    def __init__(self, flaskip, args: argparse.Namespace, cserv: ClientService, queue_man: QueueListManager) -> None:
        self.logger = logging.getLogger('web-server')
        self.flaskip = flaskip
        self.app = Flask(__name__)
        # Register CORS for the app. Regulates who can access the server
        CORS(self.app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True, expose_headers=["X-Metadata"])
        # Disable strict slashes to allow for more flexible routing (/ and no / at end treated same)        
        self.app.url_map.strict_slashes = False
        self.args = args
        self.cservice = cserv
        self.user_manager = UserManager()
        self.queue_manager = queue_man
        
        @self.app.after_request
        def after_request(response):
            """
            Handles CORS and other userful headers for all responses
            """
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Metadata')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Expose-Headers', 'X-Metadata')
            return response

                

    def run(self) -> None:
        """
        Starts the Flask server and adds routes to the server.

        TODO: Change this to work based on where Flask server starts. Also, use the actual ip address
        self.sserv.base_url = request.remote_addr
        """
        self.logger.info("Starting Flask server on %s:%s", self.flaskip, self.args.port)
        
        self.app.logger.setLevel(
            int(self.args.log)
        ) if self.args.log.isdecimal() else self.app.logger.setLevel(self.args.log)

        self.add_routes()
        self.app.run(host=self.flaskip, port=self.args.port, debug=True)

    def add_routes(self) -> None:
        """
        Creates flask routes for GET, POST, PUT, OPTION endpoints
        """
        
        @self.app.route('/routes', methods=['GET'])
        def get_routes():
            routes = []
            for rule in self.app.url_map.iter_rules():
                routes.append({
                    "endpoint": rule.endpoint,
                    "methods": list(rule.methods),
                    "path": str(rule)
                })
            return jsonify(routes)
                
        
        @self.app.route("/data/metadata/<uuid>", methods=["GET"])
        def get_nerf_metadata(uuid: str) -> Response:
            """
            Get metadata for all output types of a job
            """
            return self.cservice.get_nerf_metadata(uuid)

        @self.app.route("/data/metadata/<output_type>/<uuid>", methods=["GET"])
        def get_nerf_type_metadata(output_type: str, uuid: str) -> Response:
            """
            Get metadata for a specific output type of a job
            """
            return self.cservice.get_nerf_type_metadata(output_type, uuid)

        @self.app.route("/data/nerf/<output_type>/<uuid>", methods=["OPTIONS"])
        def preflight_nerf_resource(output_type: str, uuid: str) -> Response:
            """
            Handles preflight requests for CORS
            """
            return '', 200

        @self.app.route("/video", methods=["POST", "PUT"])
        def receive_video_post():
            """
            Receives video from POST/PUT request. Initializes job creation
            process

            Handles optional request parameters:
            Defaults to gaussian training mode and splat_cloud file generation.
            @param training_mode: gaussian, tensorf.
            @param output_types: splat_cloud, point_cloud, video, model.
            """
            self.logger.info("Received video file")

            def validate_request_params(request) -> Tuple[Optional[Response], Optional[dict]]:
                """
                Validates request parameters and returns a new training job configuration dictionary.

                TODO: Refactor this to use a object that stores valid parameters and their types instead of hardcoding them.
                
                Args:
                    request (Request): Incoming request object
                Returns:
                    Tuple[Optional[Response], Optional[dict]]: Response object and config dictionary
                    If succesfully validated, returns a new training job configuration dictionary
                """
                training_mode = request.form.get("training_mode", "gaussian")
                output_types = request.form.get("output_types", "splat_cloud,point_cloud").split(",")
                save_iterations = request.form.get("save_iterations", "700,3000")
                total_iterations = request.form.get("total_iterations", "3000")

                # Validate training mode
                valid_modes = ["gaussian", "tensorf"]
                if training_mode not in valid_modes:
                    return create_response(
                        status=NerfStatus.ERROR,
                        error=NerfError.INVALID_INPUT,
                        message=f"Invalid training mode: {training_mode}",
                        status_code=400
                    ), None
                    
                # Validate output types
                valid_types = {
                    "gaussian": ["splat_cloud", "point_cloud", "video"],
                    "tensorf": ["model", "video"]
                }
                invalid_types = [s for s in output_types if s not in valid_types[training_mode]]
                if invalid_types:
                    return create_response(
                        status=NerfStatus.ERROR,
                        error=NerfError.INVALID_INPUT,
                        message=f"Invalid output type(s) for {training_mode} mode: {', '.join(invalid_types)}",
                        status_code=400
                    ), None

                # Validate save iterations
                save_iterations = request.form.get("save_iterations", "").split(",")
                if save_iterations:
                    try:
                        save_iterations = [int(s.strip()) for s in save_iterations]
                    except ValueError:
                        return create_response(
                            status=NerfStatus.ERROR,
                            error=NerfError.INVALID_INPUT,
                            message="Invalid save iterations. Must be integer values.",
                            status_code=400
                        ), None

                # Validate total iterations
                try:
                    total_iterations = int(total_iterations)
                    if total_iterations < 0 or total_iterations > 30000:
                        raise ValueError
                except ValueError:
                    return create_response(
                        status=NerfStatus.ERROR,
                        error=NerfError.INVALID_INPUT,
                        message="Invalid total iterations. Must be an integer value in range [1,30000].",
                        status_code=400
                    ), None

                # Fallthrough to default output types. TODO might be redundant
                if not output_types:
                    if training_mode == "gaussian":
                        output_types = ["splat_cloud"]
                    elif training_mode == "tensorf":
                        output_types = ["video"]

                config = {
                    "training_mode": training_mode,
                    "output_types": output_types,
                    "save_iterations": save_iterations,
                    "total_iterations": total_iterations
                }

                return None, config
                # END VALIDATE_REQUEST_PARAMS
            
            # Check if file is present in the request
            if 'file' not in request.files:
                self.logger.error("No file part in the request")
                
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.FILE_NOT_RECEIVED,
                    message="No file part in the request",
                    status_code=400
                )
                
            # Check if filename is empty
            video_file = request.files['file']
            if video_file.filename == '':
                self.logger.error("No selected file")
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.FILE_NOT_RECEIVED,
                    message="No selected file",
                    status_code=400
                )
            # Validate request parameters
            response, config = validate_request_params(request)
            if response and not config:
                return response

            # Handle client response
            uuid = self.cservice.handle_incoming_video(video_file, config)
            if uuid is None:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.FILE_NOT_RECEIVED,
                    message="File not received or bad file extension. Valid extensions: [\".mp4\", \".wav\"]",
                    status_code=400
                )

            # Valid request, return success response
            return create_response(
                status=NerfStatus.PROCESSING,
                error=NerfError.NO_ERROR,
                message="Video received and processing. Check back later for updates.",
                uuid=uuid,
                data=config,
                status_code=200
            )

        @self.app.route("/data/nerf/<output_type>/<uuid>", methods=["GET"])
        def send_nerf_resource(output_type: str, uuid: str) -> Response:
            """
            Sends a resource of type output_type for job uuid to the client.
            Sends the resource in chunks if the client requests a range.
            You SHOULD request a range, as alot of resources can be bigger than 
            30MB (max http response size).

            Args:
                output_type (str): The type of resource to send
                uuid (str): The job id
            Returns:
                Response: 
                Request of chunked resource or error message
            """
            iteration = request.args.get("iteration", "")
            range_header = request.headers.get('Range', None)
            return self.cservice.send_nerf_resource(uuid, output_type, iteration, range_header)

        @self.app.route("/video/<vidid>", methods=["GET"])
        @deprecated("Used for legacy tensoRF front end output")
        def send_video(vidid: str) -> Response:
            """
            DEPRECATED. Serve a video to legacy web-app.

            Args:
                vidid (str): job id

            Returns:
                Response: Video content or error message
            """
            # TODO: Change routing to serve rendered videos
            try:
                if (is_valid_uuid(vidid)):
                    path = os.path.join(
                        os.getcwd(), "data/raw/videos/" + vidid + ".mp4")
                    response = make_response(
                        send_file(path, as_attachment=True))
                else:
                    response = make_response("Error: invalid UUID")
            except Exception as e:
                print(e)
                response = make_response("Error: does not exist")

            return response

        @self.app.route("/data/nerf/<vidid>", methods=["GET"])
        @deprecated("Legacy Code. Used to send rendered videos to web-app")
        def send_nerf_video(vidid: str):
            """
            LEGACY. Sends the corresponding video for job vidid to
            web-app frontend.

            Args:
                vidid (str): Job id

            Returns:
                _type_: video file content or error message
            """
            logger = logging.getLogger('web-server')
            ospath = None
            flag = 0
            status_str = "Processing"
            if is_valid_uuid(vidid):
                # If the video had no errors return the video path, otherwise return the error flag
                if flag == 0:
                    ospath = self.cservice.get_nerf_video_path(vidid)
                else:
                    flag = self.cservice.get_nerf_flag(vidid)

            if flag != 0:
                # ERROR CODE BREAKDOWN:
                # 1 = Unknown
                # 2 = File already exists
                # 3 = File not found
                # 4 = Video too blurry
                # SEE MORE IN video_to_images.py for error codes
                response = make_response(
                    "Returned with error code {}".format(flag))
            elif ospath == None or not os.path.exists(ospath):
                response = make_response(status_str)
            else:
                status_str = "Video ready"
                response = make_response(send_file(ospath, as_attachment=True))

            response.headers['Access-Control-Allow-Origin'] = '*'
            return response

        @self.app.route("/worker-data/<path:path>")
        def send_worker_data(path):
            """_summary_

            Args:
                path (_type_): _description_

            Returns:
                _type_: _description_
            """
            # serves data directory for workers to pull any local data
            # TODO make this access secure
            return send_from_directory('data', path[5:])

        @self.app.route("/login", methods=["GET"])
        def login_user():
            """
            Login a user

            Returns:
                _type_: _description_
            """
            # get username and password from login
            # use get_user_by_username and compare the password retrieved from that to the password given by the login
            # if they match allow the user to login, otherwise, fail

            username = request.form["username"]
            password = request.form["password"]

            user = self.user_manager.get_user_by_username(username)
            if user == None:
                string = f"INCORRECT USERNAME|{user.id}"
                response = make_response(string)
                return response

            if user.password == password:
                string = f"SUCCESS|{user.id}"
                response = make_response(string)
                return response
            else:
                string = f"INCORRECT PASSWORD|{user.id}"
                response = make_response(string)
                return response

        @self.app.route("/register", methods=["POST"])
        def register_user():
            """
            Register a user

            Raises:
                Exception: _description_

            Returns:
                _type_: _description_
            """
            # get username and password from register
            # use set_user
            # if it doesnt fail, youre all good

            username = request.form["username"]
            password = request.form["password"]

            user = self.user_manager.generate_user(username, password)

            if user == 1:
                string = f"USERNAME CONFLICT|{user.id}"
                response = make_response(string)
                return response
            if user == None:
                raise Exception('Unknown error when generating user')

            string = f"SUCCESS|{user.id}"
            response = make_response(string)
            return response

        @self.app.route("/queue", methods=["GET"])
        def send_queue_position(queueid: str, id: str) -> Response:
            """
            Returns the queue position of a task in a queue

            Args:
                queueid (str): "sfm_list" | "nerf_list" | "queue_list"
                id (str): uuid of task
            Returns:
                Return "Success!"
            """
            return make_response("{} / {}".format(self.queue_manager.get_queue_position(queueid, id), self.queue_manager.get_queue_size(queueid)))

        @self.app.route("/test")
        def test_endpoint():
            return "Success!"
