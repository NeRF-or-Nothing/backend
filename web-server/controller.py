import argparse
import os
from pickle import TRUE
import time
#import magic
from uuid import uuid4, UUID

from flask import Flask, request, make_response, send_file, send_from_directory, url_for, jsonify

from models.scene import UserManager, QueueListManager
from services.scene_service import ClientService
import logging

def is_valid_uuid(value):
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False

class WebServer:
    def __init__(self, flaskip, args: argparse.Namespace, cserv: ClientService, queue_man: QueueListManager) -> None:
        self.flaskip = flaskip
        self.app = Flask(__name__)
        self.args = args
        self.cservice = cserv
        self.user_manager=UserManager()
        self.queue_manager=queue_man

    def run(self) -> None:
        self.app.logger.setLevel(
            int(self.args.log)
        ) if self.args.log.isdecimal() else self.app.logger.setLevel(self.args.log)

        self.add_routes()
        
        # TODO: Change this to work based on where Flask server starts. Also, use the actual ip address
        ### self.sserv.base_url = request.remote_addr

        self.app.run(host=self.flaskip,port=self.args.port)

    def add_routes(self) -> None:

        #TODO: Write error handling so the whole server doesn't crash when the user sends incorrect data.
        @self.app.route("/video", methods=["POST", "PUT"])
        def recv_video():
            """
            Posts 
            
            Handles optional request parameters:
            Defaults to gaussian training mode and splat file generation.
            @param training_mode: gaussian, tensorf.
            @param output_types: splat, ply, video, model.
            """
            
            video_file = request.files.get("file")
            
            # TODO: Add support for user parameters
            
            # Handle optional request paramters
            training_mode = request.form.get("training_mode", "gaussian").lower()
            output_types_str = request.form.get("output_types", "").lower()
            output_types = [s.strip() for s in output_types_str.split(",") if s != ""]
            
            # Validate optional training mode
            valid_modes = ["gaussian, tensorf"] 
            if training_mode not in valid_modes:
                return make_response(jsonify({
                    "error" : f"Invalid training mode: {training_mode}",
                    "valid_modes" : valid_modes
                }), 400)
            
            # Validate optional output type
            valid_types = {
                "gaussian" : ["splat", "ply", "video"],
                "tensorf" : ["model", "video"]
            }
            
            invalid_types = [s for s in output_types if s not in valid_types[training_mode]]
            if invalid_types:
                return make_response(jsonify({
                    "error" : f"Invalid output type(s) for {training_mode} mode : {', '.join(invalid_types)}",
                    "valid_types" : valid_types[training_mode]
                }), 400)
            
            # Fallthrough to default output types  
            if not output_types:
                if training_mode == "gaussian":
                    output_types = ["splat"]
                elif training_mode == "tensorf":
                    output_types = ["video"]
            
            # TODO: UUID4 is cryptographically secure on CPython, but this is not guaranteed in the specifications.
            # Might want to change this.
            # TODO: Don't assume videos are in mp4 format
            uuid, output_paths = self.cservice.handle_incoming_video(video_file, training_mode, output_types)
            if(uuid is None):
                response = make_response("ERROR")
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            
            # TODO: now pass to nerf/tensorf/colmap/sfm, and decide if synchronous or asynchronous
            # will we use a db for cookies/ids?
                
            response = make_response(jsonify({
                "uuid" : uuid,
                "training_mode" : training_mode,
                "output_paths" : output_paths
            }))
            response.headers['Access-Control-Allow-Origin'] = '*'

            return response

        @self.app.route("/video/<vidid>", methods=["GET"])
        def send_video(vidid: str):
            # TODO: Change routing to serve rendered videos
            try:
                if(is_valid_uuid(vidid)):
                    path = os.path.join(os.getcwd(), "data/raw/videos/" + vidid + ".mp4")
                    response = make_response(send_file(path, as_attachment=True))
                else:
                    response = make_response("Error: invalid UUID")
            except Exception as e:
                print(e)
                response = make_response("Error: does not exist")
           
            return response
            
        @self.app.route("/data/nerf/splat/<splatid>", methods=["GET"])
        def send_nerf_splat(splatid: str):
            logger = logger.getLogger('webs-server')
            ospath = None
            flag = 0
            status_str = "Processing"
            if is_valid_uuid(splatid):
                # If the splat file generation had no errors return splat path or else error flag
                if flag == 0:
                    ospath = self.cservice.get_nerf_splat_path(splatid)
                else:
                    flag = self.cservice.get_nerf_flag(splatid)
                    
            if flag != 0:
                # ERROR CODE BREAKDOWN:
                # 1 = Unknown
                # 2 = File already exists
                # 3 = File not found
                # 4 = Video too blurry
                # SEE MORE IN video_to_images.py for error codes
                response = make_response("Returned with error code {}".format(flag))
            elif ospath == None or not os.path.exists(ospath):
                response = make_response(status_str)
            else:
                status_str = "Splat file ready"
                response = make_response(send_file(ospath, as_attachment=True))
            response.headers["Access-Control-Allow-Origin"] = '*'
            return response
                
            
        @self.app.route("/data/nerf/<vidid>", methods=["GET"])
        def send_nerf_video(vidid: str):
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
                response = make_response("Returned with error code {}".format(flag))
            elif ospath == None or not os.path.exists(ospath):
                response = make_response(status_str)
            else:
                status_str = "Video ready"
                response = make_response(send_file(ospath, as_attachment=True))
                
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response

        @self.app.route("/worker-data/<path:path>")
        def send_worker_data(path):
            # serves data directory for workers to pull any local data
            # TODO make this access secure
            return send_from_directory('data',path[5:])
            
        @self.app.route("/login", methods=["GET"])
        def login_user():
            #get username and password from login 
            #use get_user_by_username and compare the password retrieved from that to the password given by the login
            #if they match allow the user to login, otherwise, fail

            username=request.form["username"]
            password=request.form["password"]

            user=self.user_manager.get_user_by_username(username)
            if user==None:
                string=f"INCORRECT USERNAME|{user.id}"
                response=make_response(string)
                return response

            if user.password == password:
                string=f"SUCCESS|{user.id}"
                response=make_response(string)
                return response
            else:
                string=f"INCORRECT PASSWORD|{user.id}"
                response=make_response(string)
                return response



        @self.app.route("/register", methods=["POST"])
        def register_user():
            #get username and password from register
            #use set_user
            #if it doesnt fail, youre all good


            username=request.form["username"]
            password=request.form["password"]

            user=self.user_manager.generate_user(username,password)

            if user==1:
                string=f"USERNAME CONFLICT|{user.id}"
                response=make_response(string)
                return response
            if user==None:
                raise Exception('Unknown error when generating user')



            string=f"SUCCESS|{user.id}"
            response=make_response(string)
            return response

        # Returns the queue position of a task in a queue
        # queueid: sfm_list, nerf_list, queue_list
        # id: uuid of task
        @self.app.route("/queue",methods=["GET"])
        def send_queue_position(queueid: str, id: str):
            return make_response("{} / {}".format(self.queue_manager.get_queue_position(queueid,id),self.queue_manager.get_queue_size(queueid)))

        @self.app.route("/test")
        def test_endpoint():
            
            return "Success!"

