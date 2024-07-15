"""
This file loads the environment variables from the .env file.
Afterwards, the Web Server is started.
"""

from argparser import create_arguments
from controller import WebServer
import threading
from models.managers import SceneManager, QueueListManager
from services.queue_service import RabbitMQService, digest_finished_sfms, digest_finished_nerfs, RabbitMQServiceV2
from services.scene_service import ClientService
from services.clean_service import cleanup
from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv

import logging
from log import web_server_logger

# TODO: Add sphinx documentation generation. Probably use git actions to auto generate docs on push to master
def main():
    """
     STARTING LOGGER
    """
    logger = web_server_logger('web-server')
    logger.info("~WEB SERVER~")

    logger.info("Starting web-app...")
    
    parser = create_arguments()
    args = parser.parse_args()

    ipfile = open(args.configip)
    #docker_in.json inside docker container
    #docker_out.json outside docker container
    ipdata = json.load(ipfile)
    
    # Load environmental 
    load_dotenv()
             
    rabbitip = str(os.getenv("RABBITMQ_IP"))
    flaskip = ipdata["flaskdomain"]

    # Shared Database manager <from models>
    # SceneManager shared across threads since it is thread safe
    scene_man = SceneManager()

    # QueueListManager to manage list positions,shared
    queue_man = QueueListManager()
    
    # Rabbitmq service to post/consume jobs to/from the workers <from services>
    rmq_service = RabbitMQServiceV2(rabbitip, queue_man, scene_man)

    # TODO: async worker to clean up old data
    
    # service to handle all incoming client requests from the controller <from services>
    c_service = ClientService(scene_man, rmq_service)

    # start listening to incoming requests on the controller <from controllers>
    server = WebServer(flaskip, args, c_service, queue_man)

    ipfile.close() 
    server.run()

if __name__ == "__main__":
    main()
    
