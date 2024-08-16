import requests
import pika
import json
import time
import os
import cv2
import numpy as np

import logging
from image_position_extractor import extract_position_data
from log import sfm_worker_logger
from video_to_images import split_video_into_frames
from colmap_runner import run_colmap
from matrix import get_json_matrices
from opt import config_parser

from flask import Flask
from flask import send_from_directory
from pathlib import Path
from multiprocessing import Process
from dotenv import load_dotenv
from requests.packages.urllib3.exceptions import InsecureRequestWarning

app = Flask(__name__)
base_url = "http://sfm-worker:5100/"
cert_path = '/app/secrets/cert.pem' # Local Self-Signed Cert Path


@app.route("/data/outputs/<path:path>")
def send_video(path):
    logger.info(f"Sending video: {path}")
    return send_from_directory("data/outputs/", path)


def start_flask():
    global app
    app.run(host="0.0.0.0", port=5100, debug=False)


def to_url(local_file_path: str):
    return base_url.rstrip('/') + '/' + local_file_path.lstrip('/')


def is_background_white(video_file_path, threshold=0.9, sample_frames=10):
    """
    Determine if the video background is predominantly white.
    
    Args:
        video_file_path (str): Path to the video file
        threshold (float): Threshold for considering a pixel as white (0-1)
        sample_frames (int): Number of frames to sample from the video
    Returns:
        bool: True if the background is considered white, False otherwise
    """
    logger = logging.getLogger('sfm-worker')
    
    cap = cv2.VideoCapture(video_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // sample_frames
    
    white_pixel_ratio = 0
    
    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Count white pixels
        white_pixels = np.sum(gray > 255 * threshold)
        total_pixels = gray.size
        
        white_pixel_ratio += white_pixels / total_pixels
    
    cap.release()
    
    # Calculate average white pixel ratio
    avg_white_ratio = white_pixel_ratio / sample_frames
    
    logger.info(f"Average white pixel ratio: {avg_white_ratio}")
    
    return bool(avg_white_ratio > 0.5)  # Consider background white if more than 50% of pixels are white



def run_full_sfm_pipeline(id, video_file_path, input_data_dir, output_data_dir):
    # run colmap and save data to custom directory
    # Create output directory under data/output_data_dir/id
    # TODO: use library to fix filepath joining
    if not output_data_dir.endswith(("\\", "/")) and not id.startswith(("\\", "/")):
        output_data_dir = output_data_dir + "/"
    output_path = output_data_dir + id
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)

    # Get logger
    logger = logging.getLogger('sfm-worker')

    # Check if the background is white
    is_white_background = is_background_white(video_file_path)
    logger.info(f"Video background is {'white' if is_white_background else 'not white'}")

    # (1) vid_to_images.py
    imgs_folder = os.path.join(output_path, "imgs")
    logger.info("Video file path:{}".format(video_file_path))

    split_status = split_video_into_frames(video_file_path, imgs_folder, 100)
    # Catches flag for blurriness
    if split_status == 4:
        logger.error("Video is too blurry.")
        # motion_data flag option determines the status of the job
        # flag = 4 means the video was too blurry
        motion_data = {"flag": 4, "id": id}
        return motion_data, None

    # imgs are now in output_data_dir/id

    # (2) colmap_runner.py
    colmap_path = "/usr/local/bin/colmap"
    status = run_colmap(colmap_path, imgs_folder, output_path)
    if status == 0:
        logger.info("COLMAP ran successfully.")
    elif status == 1:
        logger.error("ERROR: There was an unknown error running COLMAP")

    # (3) matrix.py
    initial_motion_path = os.path.join(output_path, "images.txt")
    camera_stats_path = os.path.join(output_path, "cameras.txt")
    parsed_motion_path = os.path.join(output_path, "parsed_data.csv")

    extract_position_data(initial_motion_path, parsed_motion_path)
    motion_data = get_json_matrices(camera_stats_path, parsed_motion_path)
    motion_data["id"] = id
    motion_data["flag"] = 0
    motion_data["white_background"] = is_white_background
    
    print("motion_data ", motion_data, flush=True)

    # Save copy of motion data
    with open(os.path.join(output_path, "transforms_data.json"), "w") as outfile:
        outfile.write(json.dumps(motion_data, indent=4))

    return motion_data, imgs_folder


def colmap_worker():
    load_dotenv()
    input_data_dir = "data/inputs/"
    output_data_dir = "data/outputs/"
    Path(f"{input_data_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_data_dir}").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('sfm-worker')

    def process_colmap_job(ch, method, properties, body):
        logger.info("Starting New Job")
        logger.info(body.decode())
        job_data = json.loads(body.decode())
        id = job_data["id"]
        
        logger.info(f"Running New Job With ID: {id}")

        # TODO: Handle exceptions and enable steaming to make safer
        job_data["file_path"] = job_data["file_path"].replace('host.docker.internal', 'web-server')
        print("job_data", job_data, flush=True)
        video = requests.get(job_data["file_path"], timeout=10, verify=cert_path)
        logger.info("Web server pinged")
        video_file_path = f"{input_data_dir}{id}.mp4"
        logger.info(f"Saving video to: {video_file_path}")
        open(video_file_path, "wb").write(video.content)
       
        logger.info("Video downloaded")

        # RUNS COLMAP AND CONVERSION CODE
        motion_data, imgs_folder = run_full_sfm_pipeline(
            id, video_file_path, input_data_dir, output_data_dir
        )
        # Catch incomplete videos by flag != 1 and return here
        if motion_data["flag"] != 0:
            logger.error("An error was found. Ending process.")
            channel.basic_publish(
            exchange="", routing_key="sfm-out", body=json.dumps(motion_data)
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        
        # create links to local data to serve
        for i, frame in enumerate(motion_data["frames"]):
            file_name = frame["file_path"]
            file_path = os.path.join(imgs_folder, file_name)
            file_url = to_url(file_path)
            motion_data["frames"][i]["file_path"] = file_url

        json_motion_data = json.dumps(motion_data)
        channel.basic_publish(
            exchange="", routing_key="sfm-out", body=json_motion_data
        )

        # confirm to rabbitmq job is done
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Job complete")


    rabbitmq_domain = "rabbitmq"
    credentials = pika.PlainCredentials(
        str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(
        rabbitmq_domain, 5672, '/', credentials, heartbeat=300
    )

    # retries connection until connects or 2 minutes pass
    timeout = time.time() + 60 * 2
    while True:
        if time.time() > timeout:
            raise Exception(
                "nerf_worker took too long to connect to rabbitmq")
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue='sfm-in')
            channel.queue_declare(queue='sfm-out')

            # Will block and call process_nerf_job repeatedly
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue='sfm-in', on_message_callback=process_colmap_job, auto_ack=False)
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
                break
        except pika.exceptions.AMQPConnectionError:
            continue
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="sfm-in", on_message_callback=process_colmap_job)
    channel.start_consuming()
    logger.critical("Should not get here: After consumption of RabbitMQ.")

if __name__ == "__main__":  
    input_data_dir = "data/inputs/"
    output_data_dir = "data/outputs/"
    Path(f"{input_data_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_data_dir}").mkdir(parents=True, exist_ok=True)

    """
     STARTING LOGGER
    """
    logger = sfm_worker_logger('sfm-worker')
    logger.info("~SFM WORKER~")

    # Disable SSL verification warnings
    # TODO IMPORTANT - Replace with a proper SSL certificate in deployment
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    # def test_web_server_connection():
    #     print(f"Certificate path: {cert_path}")
    #     print(f"Certificate exists: {os.path.exists(cert_path)}")
        
    #     try:
    #         response = requests.get("https://web-server:5000/health", timeout=5, verify=cert_path)
    #         response.raise_for_status()
    #         print(f"Successfully connected to web-server via HTTPS. Response: {response.text}")
    #     except requests.exceptions.RequestException as e:
    #         print(f"Failed to connect to web-server via HTTPS: {str(e)}")
    #         print(f"SSL Version: {ssl.OPENSSL_VERSION}")
            
    #         # Try to establish a raw SSL connection
    #         try:
    #             context = ssl.create_default_context(cafile=cert_path)
    #             with socket.create_connection(("web-server", 5000)) as sock:
    #                 with context.wrap_socket(sock, server_hostname="web-server") as secure_sock:
    #                     print(f"SSL connection established. Peer certificate: {secure_sock.getpeercert()}")
    #         except Exception as ssl_e:
    #             print(f"Raw SSL connection failed: {str(ssl_e)}")


    # test_web_server_connection()

    # Load args from config file
    args = config_parser()

    # Local run behavior
    if args.local_run == True:
        motion_data, imgs_folder = run_full_sfm_pipeline(
            "Local_Test", args.input_data_path, input_data_dir, output_data_dir
        )
        logger.info("MOTION DATA: {}".format(motion_data))
        json_motion_data = json.dumps(motion_data)

    # Standard webserver run behavior
    else:
        sfmProcess = Process(target=colmap_worker, args=())
        flaskProcess = Process(target=start_flask, args=())
        flaskProcess.start()
        sfmProcess.start()
        flaskProcess.join()
        sfmProcess.join()
