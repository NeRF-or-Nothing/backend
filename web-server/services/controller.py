"""
This module contains the WebServer class, which is responsible for handling incoming requests to the Flask server portion of web-server.
It is the main controller for the web-server, delegating tasks to the ClientService and QueueListManager classes.
Additionally handling worker data requests, user login and registration, and queue position requests.

NOTE TO ALL FUTURE DEVELOPERS:
    - Before deploying, we need a reverse proxy that will handle actual SSL implementation. This allows for all external requests to be HTTPS,
      and internal requests to be HTTP
    - This module should probably be broken down to its core, and any functionality but route handling and JWT (maybe even this goes to reverse proxy)
      should be moved to the ClientService class
    - Currently, Docker Compose generates a self-signed certificate that is stored in a shared volume across all workers. When implementing a reverse proxy,
      you will have to modify communication on worker-side restrictions so workers can only communicate with webserver
    - I tried very hard to get HTTPS and HTTP to work in tandem, and it was a nightmare. I would recommend just using HTTP for everything until deployment
    
TODO: Unify error handling and response messages
TODO: Create endpoint to list all available resources for a job id. Should prob follow call stack of get user data -> get all user jobs -> get resource data for job
TODO: REJECTED Probably could add api key authentication for those resources, bc it should be quite easy to get user by api_key
TODO: DONE 8/7/24 Look into bcrypt for handling, dont have to store salt then
TODO: DONE 8/7/24 Create authorization scheme. Thinking use https for POST /login, which returns a JSON Web Token with expiration, it will allow frontend to never access api key directly, let frontend be stateless, and only expose user id to frontend
TODO: DONE 8/7/24 Implement secure password handling and storage.
TODO: WIP  8/7/24 Add actual use for User class. Maybe training minutes? (User Scene History?)
TODO: STRETCH. Create token expiry. They should be short lived, and refreshed on each request. Currently are unlimited per session.
"""


import logging
import argparse
import os
from pathlib import Path

from models.managers import QueueListManager
from models.status import NerfError, NerfStatus, UserStatus, UserError
from models.scene import NerfV2
from utils.response_utils import create_response
from services.client_service import ClientService

from dotenv import load_dotenv
from typing import Optional, Tuple
from typing_extensions import deprecated
from uuid import UUID
from functools import wraps
from flask_cors import CORS
from flask_sslify import SSLify
import os
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from flask import (
    Flask,
    request,
    make_response,
    send_file,
    send_from_directory,
    jsonify,
    Response,
    redirect,
    url_for
)


class WebServer:
    """
    The WebServer class is responsible for handling incoming requests to the Flask server portion of web-server.
    The only processing should be request validation, and web token handling. Delegates tasks to the ClientService and QueueListManager classes.

    TODO: DONE 7/8/24 Clean this up a lot.
    TODO: Use real ssl cert instead of gen'd development one
    """

    def __init__(self, flaskip, jwt_key, args: argparse.Namespace, cserv: ClientService, queue_man: QueueListManager) -> None:
        self.logger = logging.getLogger('web-server')
        self.flaskip = flaskip
        self.app = Flask(__name__)
        self.args = args
        self.client_service = cserv
        self.queue_manager = queue_man

        @self.app.after_request
        def after_request(response):
            """
            Handles CORS headers and other userful headers for all responses
            """
            response.headers.add('Access-Control-Allow-Headers',
                                 'Content-Type,Authorization,X-Metadata')
            response.headers.add(
                'Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            response.headers.add('Access-Control-Expose-Headers', 'X-Metadata')
            return response

        # Setup CORS to only accept from localhost:3000
        # CORS(self.app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5100", "http://localhost:5200"]}}, supports_credentials=True, expose_headers=["X-Metadata"])
        CORS(self.app, resources={
             r"/*": {"origins":
                     [
                         "http://localhost:3001",
                         "http://localhost:5173"
                     ]
                     }
             },
             supports_credentials=True,
             expose_headers=["X-Metadata"]
             )
        # Handle extra '/' in routes
        self.app.url_map.strict_slashes = False

        # Force HTTPS for requests
        self.sslify = SSLify(self.app, permanent=True)

        # Setup JWT for user authentication
        self.app.config["JWT_SECRET_KEY"] = jwt_key
        self.jwt = JWTManager(self.app)

        @self.app.before_request
        def before_request():
            print("Received request from", request.remote_addr, flush=True)

            # Docker's default subnet
            if request.remote_addr.startswith('172.'):
                self.app.config['PREFERRED_URL_SCHEME'] = 'http'
            else:
                self.app.config['PREFERRED_URL_SCHEME'] = 'https'

    def run(self) -> None:
        """
        Starts the Flask server and adds routes to the server.

        TODO: Change this to work based on where Flask server starts. Also, use the actual ip address
        self.sserv.base_url = request.remote_addr
        """
        self.logger.info("Starting Flask server on %s:%s",
                         self.flaskip, self.args.port)

        self.app.logger.setLevel(
            int(self.args.log)
        ) if self.args.log.isdecimal() else self.app.logger.setLevel(self.args.log)

        self.setup_routes()

        # Start the Flask server with SSL context
        self.app.run(
            host=self.flaskip,
            port=self.args.port,
            ssl_context=('secrets/cert.pem', 'secrets/key.pem'),
            debug=True
        )

    def setup_routes(self) -> None:
        """
        Creates flask routes for GET, POST, PUT, OPTION endpoints
        """
        self.app.route("/video",
                       methods=["POST", "PUT"])(self.receive_video)
        self.app.route("/login",
                       methods=["POST"])(self.login_user)
        self.app.route("/register",
                       methods=["POST"])(self.register_user)
        self.app.route('/routes',
                       methods=['GET'])(self.send_routes)
        self.app.route("/data/metadata/<uuid>",
                       methods=["GET"])(self.send_nerf_metadata)
        self.app.route("/data/metadata/<output_type>/<uuid>",
                       methods=["GET"])(self.send_nerf_type_metadata)
        self.app.route("/data/nerf/<output_type>/<uuid>",
                       methods=["GET"])(self.send_nerf_resource)
        self.app.route("/worker-data/<path:path>",
                       methods=["GET"])(self.send_to_worker)
        self.app.route("/queue",
                       methods=["GET"])(self.send_queue_position)
        self.app.route("/history",
                       methods=["GET"])(self.send_user_history)
        self.app.route("/preview/<uuid>",
                       methods=["GET"])(self.send_preview)
        self.app.route("/video/<vidid>",
                       methods=["GET"])(self.send_video)
        self.app.route("/data/nerf/<vidid>",
                       methods=["GET"])(self.send_nerf_video)
        self.app.route("/health"
                       )(self.health_check)

    @staticmethod
    def token_required(f):
        """
        Decorator that verifies a valid JWT token is included in the request headers.
        The JWT token should contain the encoded user's id. By default, valid requests
        will return the wrapped function with additional arguments of user id
        inserted as first argument so that the client service can validate the
        users access to the requested resource.
        """
        @wraps(f)
        @jwt_required()
        def decorated(self, *args, **kwargs):
            try:
                user_id = get_jwt_identity()
                assert (self.client_service.user_manager.get_user_by_id(
                    user_id) != UserError.USER_NOT_FOUND)
            except:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=UserError.INVALID_JWT,
                    message="Invalid JWT",
                    status_code=401
                )
            return f(self, user_id, *args, **kwargs)
        return decorated

    @token_required
    def send_nerf_metadata(self, user_id: str, uuid: str) -> Response:
        """
        Get metadata for all output types of a job
        """
        print("Getting metadata for job", uuid, flush=True)
        return self.client_service.get_nerf_metadata(user_id, uuid)

    @token_required
    def send_nerf_type_metadata(self, user_id: str, output_type: str, uuid: str) -> Response:
        """
        Get metadata for a specific output type of a job
        """
        print("Getting metadata for job", uuid,
              "output type", output_type, flush=True)
        return self.client_service.get_nerf_type_metadata(user_id, uuid, output_type)

    @token_required
    def send_nerf_resource(self, user_id: str, output_type: str, uuid: str) -> Response:
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
        print("Sending resource", output_type, "for job", uuid, flush=True)

        iteration = request.args.get("iteration", "")
        range_header = request.headers.get('Range', None)
        return self.client_service.send_nerf_resource(user_id, uuid, output_type, iteration, range_header)

    @token_required
    def send_user_history(self, user_id: str) -> Response:
        """
        Get the history of a user's jobs

        Args:
            user_id (str): The user's id
        Returns:
            Response:
            User's job history or error message
        """
        print("Getting user history for user", user_id, flush=True)
        return self.client_service.get_user_history(user_id)

    def send_to_worker(self, path):
        """
        Sends data to workers at start of each training stage
        e.g. Gaussian training requesting images and sfm transforms.
        """
        print("Sending worker data from path: ", path, flush=True)
        self.logger.info(f"Attempting to send worker data from path: {path}")
        self.logger.info(f"to path: {request.remote_addr}: {request.environ.get('REMOTE_PORT')}")

        if not os.path.exists(path):
            self.logger.error(f"File not found: {path}")
            return make_response("File not found", 404)

        try:
            return send_from_directory(os.path.dirname(os.path.abspath(path)), os.path.basename(path))
        except Exception as e:
            self.logger.error(f"Error sending worker data: {str(e)}")
            return make_response(f"Error: {str(e)}", 500)

    @token_required
    def send_preview(self, user_id: str, uuid: str):
        """
        Sends preview of completed job to user, as scene name and preview image
        """
        return self.client_service.get_preview(user_id, uuid)

    @token_required
    def receive_video(self, user_id: str):
        """
        Receives video from POST/PUT request. Initializes job creation
        process. Handles optional form parameters for training config.

        TODO: This breaks srp for WebServer. Should be split into multiple functions and/or moved to cservice
        """
        self.logger.info("Received video file")

        def validate_request_params(request) -> Tuple[Optional[Response], Optional[dict]]:
            """
            Validates request parameters and returns a new 
            training job configuration dictionary.

            TODO: DONE 8/6/24 Refactor this to use a object that stores valid \
            parameters and their types instead of hardcoding them.

            Args:
                request (Request): Incoming request object
            Returns:
                Tuple[Optional[Response], Optional[dict]]: Response object and config dictionary
                If succesfully validated, returns a new training job configuration dictionary
            """
            training_mode = request.form.get("training_mode", "gaussian")
            output_types = request.form.get("output_types", "").split(",")
            save_iterations = request.form.get(
                "save_iterations", "700,3000").split(",")
            total_iterations = request.form.get("total_iterations", "3000")

            if not NerfV2.is_valid_training_mode(training_mode):
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.INVALID_INPUT,
                    message=f"Invalid training mode: {training_mode}",
                    status_code=400
                ), None

            invalid_types = [
                t for t in output_types if not NerfV2.is_valid_output_type(training_mode, t)]
            if invalid_types and output_types != [""] and output_types != []:
                return create_response(
                    status=NerfStatus.ERROR,
                    error=NerfError.INVALID_INPUT,
                    message=f"Invalid output type(s) for {training_mode} \
                        mode: {', '.join(invalid_types)}",
                    status_code=400
                ), None

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

            if not output_types:  # Probably Redundant
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

        if 'file' not in request.files:
            self.logger.error("No file part in the request")
            return create_response(
                status=NerfStatus.ERROR,
                error=NerfError.FILE_NOT_RECEIVED,
                message="No file part in the request",
                status_code=400
            )

        video_file = request.files['file']
        if video_file.filename == '':
            self.logger.error("No selected file")
            return create_response(
                status=NerfStatus.ERROR,
                error=NerfError.FILE_NOT_RECEIVED,
                message="No selected file",
                status_code=400
            )

        response, config = validate_request_params(request)
        if response and not config:
            return response

        print("Config ", config, flush=True)

        scene_name = request.form.get("scene_name")
        uuid = self.client_service.handle_incoming_video(
            user_id, video_file, config, scene_name)

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

    def login_user(self):
        """
        Login a user. Redirects http to https for security.
        TODO: Consider moving actual implementation to ClientService

        Returns:
            Response: Response containing JWT token for session or error
            message
        """
        print("Logging in user", flush=True)

        if not request.is_secure:
            return redirect(url_for('login_user', _external=True, _scheme='https'))

        username = request.form["username"]
        password = request.form["password"]

        if not username or not password:
            return create_response(
                status=UserStatus.ERROR,
                error=UserError.INVALID_PASSWORD,
                message="Username or password not provided",
                status_code=400
            )

        user = self.client_service.user_manager.get_user_by_username(
            username)

        if user == UserError.USER_NOT_FOUND:
            return create_response(
                status=UserStatus.ERROR,
                error=UserError.INVALID_USERNAME,
                message="Incorrect username",
                status_code=401
            )

        if user.check_password(password):
            access_token = create_access_token(identity=user._id)

            return create_response(
                status=UserStatus.SUCCESS,
                error=UserError.NO_ERROR,
                message="Login successful",
                data={"jwtToken": access_token},
                status_code=200
            )
        else:
            return create_response(
                status=UserStatus.ERROR,
                error=UserError.INVALID_PASSWORD,
                message="Incorrect password",
                status_code=401
            )

    def register_user(self) -> Response:
        """
        Register a user. Redirects http to https for security.

        TODO: DONE 7/26/24 Use secure password handling
        TODO: Consider moving actual implementation to ClientService

        Returns:
            Response: Response containing success message or error message
        """
        self.logger.info("Registering user")

        print("Registering user", flush=True)

        if not request.is_secure:
            return redirect(url_for('register_user', _external=True, _scheme='https'))

        username = request.form["username"]
        password = request.form["password"]

        self.logger.info("Registering user %s, pw %s", username, password)
        status, error = self.client_service.user_manager.generate_user(
            username, password)

        if status == UserStatus.ERROR:
            if error == UserError.USERNAME_ALREADY_EXISTS:
                return create_response(
                    status=status,
                    error=error,
                    message="Username already exists",
                    status_code=409
                )
            else:
                return create_response(
                    status=status,
                    error=error,
                    message="An unknown error occurred",
                    status_code=500
                )

        return create_response(
            status=NerfStatus.SUCCESS,
            error=NerfError.NO_ERROR,
            message="User registration successful",
            status_code=201)

    def send_queue_position(self, queueid: str, id: str) -> Response:
        """
        Returns the queue position of a task in a queue

        Args:
            queueid (str): "sfm_list" | "nerf_list" | "queue_list"
            id (str): uuid of task
        Returns:
            Return "Success!"
        """
        return make_response("{} / {}".format(self.queue_manager.get_queue_position(queueid, id), self.queue_manager.get_queue_size(queueid)))

    def send_routes(self) -> Response:
        """
        Lists all available routes for the server=
        """
        routes = []
        for rule in self.app.url_map.iter_rules():
            routes.append({
                "endpoint": rule.endpoint,
                "methods": list(rule.methods),
                "path": str(rule)
            })
        return jsonify(routes)

    def health_check(self):
        return "OK", 200

    @deprecated("Used for legacy tensoRF vue front_end end")
    def send_video(self, vidid: str) -> Response:
        """
        LEGACY. Serve a video to legacy web-app.

        Args:
            vidid (str): job id

        Returns:
            Response: Video content or error message
        """
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

    @deprecated("Legacy Code. Used to send rendered videos to vue web-app")
    def send_nerf_video(self, vidid: str):
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
                ospath = self.client_service.get_nerf_video_path(vidid)
            else:
                flag = self.client_service.get_nerf_flag(vidid)

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


@deprecated("Legacy Code, used for old tensoRF api")
def is_valid_uuid(value):
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False
