"""
This file contains output generation utilities for gaussian splatting and overall vidtonerf api formatting

TODO: Maybe remove kwargs stuff. More confusing than not
"""

import render
from utils.nerf_utils import convert_ply_to_splat

import logging
from pathlib import Path


def generate_output_ply(output_dict: dict, base_url: str, output_path: Path, save_iterations: list):
    """
    Handles .ply output file generation. .ply files are 
    already generated for each snapshot interval given by
    train.py. Appends to output_dict the proper api
    endpoints to probe to retrieve these files

    Args:
        output_dict (dict): data needed by web-server
        to handle complete nerf job. Must contain "id" key.
        base_url (Path): root path for worker
        output_path (Path): relative path to this jobs output directory
        save_iterations (list[int]): list of snapshot iterations to save
    Raises:
        Exception: If no iterations are provided to save video outputs
    """
    logger = logging.getLogger("nerf-worker-gaussian")
    
    id = output_dict["id"]
    if not (save_iterations or id):
        logger.warning("No id or iterations provided to save point_cloud outputs. Job: %s", id)
        raise Exception("No id or iterations provided to save point_cloud outputs")

    # Populate endpoint
    output_dict["output_endpoints"]["point_cloud"]["save_iterations"] = save_iterations
    logger.info("Generated and populated all point clouds. Job: %s", id)


def generate_output_splat(output_dict: dict, base_url: str, output_path: Path, save_iterations: list):
    """
    Handles generation of compressed .splat file from .ply. 
    Appends to output_dict the proper api endpoints to probe
    to retrieve these files

    TODO: DONE 7/10/24 Add support for **kwargs to contain which snapshots to convert, currently 30000

    Args:
        output_dict (dict): data needed by web-server
        to handle complete nerf job. Must contain "id" key.
        base_url (str): root url for worker
        output_path (Path): relative path to this jobs output directory
        save_iterations (list[int]): list of snapshot iterations to save
    Raises:
        Exception: If no iterations are provided to save video outputs
    """
    logger = logging.getLogger("nerf-worker-gaussian")

    id = output_dict["id"]
    logger.debug(f"Id: {id}, save_iterations: {save_iterations}", flush=True)
    if not (save_iterations or id):
        logger.warning("No id or iterations provided to save splat cloud outputs. Job: %s", id)
        raise Exception("No id or iterations provided to save splat cloud outputs")

    # Write splat files
    for iteration in save_iterations:
        ply_file_path = output_path / f"point_cloud/iteration_{iteration}/{id}.ply"
        splat_bytes = convert_ply_to_splat(ply_file_path, 10_000_000)
        splat_file_path = ply_file_path.parent / f"{id}.splat"
        open(splat_file_path, "wb").write(splat_bytes)
    
    # Populate endpoint
    output_dict["output_endpoints"]["splat_cloud"]["save_iterations"] = save_iterations
    logger.info("Generated and populate all splat clouds. Job: %s", id)


def generate_output_video(output_dict: dict, base_url: str, output_path: Path, save_iterations: list, **kwargs):
    """
    Handles generation of video following cameras as given by 
    output_path/cameras.json. Appends the proper api endpoint to probe
    to retrieve these files.

    TODO: DONE 7/10/24 Add support for **kwargs to contain additional params such as resolution
    TODO: Unify render output with render.py. Currently saves all frames as .png
    TODO: DEPRECATED Switch to .mp4 output and better output folder handling
    TODO: Verify that this function works as intended

    Args:
        output_dict (dict): data needed by web-server
        to handle complete nerf job. Must contain "id" key. 
        base_url (Path): root url for worker
        output_path (Path): relative path to this jobs output directory
        save_iterations (list[int]): list of snapshot iterations to save
        **kwargs (dict): additional arguments such as save iterations and resolution
    Raises:
        Exception: If no iterations are provided to save video outputs
    """
    logger = logging.getLogger("nerf-worker-gaussian")

    resolution_width = kwargs["resolution_width"]
    id = output_dict["id"]
    if not (save_iterations or id):
        logger.warning("No id or iterations provided to save video outputs. Job: %s", id)
        raise Exception("No id or iterations provided to save video outputs")
        
    # Render videos and populate endpoints
    for iteration in save_iterations:
        render_args = [
            "--model_path", str(output_path),
            "--iteration", str(iteration),
            "--skip_test"
        ]
        if resolution_width:
            render_args.extend(["--resolution_width", str(resolution_width)])
            
        render.main(render_args)

    # Populate endpoint
    output_dict["output_endpoints"]["video"]["save_iterations"] = save_iterations
    logger.info("Generated and populate all videos. Job: %s", id)


def populate_outputs(output_dict: dict, types: list, base_url: str, output_path: Path, **kwargs):
    """
    Handles generation of each unique file type that can be 
    created by gaussian splatting. Populates output_dict to contain
    appropriate GET endpoints to retrieve outputs

    TODO: DONE 8/10/24 Remove hardcoded base_url. Here bc some funky stuff with Path normalizing "//" in url even though it shouldnt

    Args:
        output_dict (dict): data needed by web-server to handle completed nerf job.
        Must contain "id" key.
        types (str): types of output files to generate
        base_url (str): root url for worker
        output_path (Path): relative path to this jobs output directory
        **kwargs (dict) : Additional file generation configuration. For instance which
        snapshot iterations to convert from .ply to .splat format or video resolution

        `output_dict` should at MINIMUM be
        output_dict = {
            "id" : str(uuid4),
            "output_endpoints" : dict[str, list[str]] # Lists of endpoints as items
        }

        `kwargs` should at MINIMUM be 
        kwargs = {
            "save_iterations":  list[int]
        }
    """
    logger = logging.getLogger("nerf-worker-gaussian")
    save_iterations = kwargs["save_iterations"]
    
    for type in types:
        type_endpoint = f"{base_url}{output_path.parent}/{type}/{output_dict['id']}" # Intentionally missing "/"

        output_dict["output_endpoints"][type] = {
            "endpoint": str(type_endpoint),
            "save_iterations": []
        }

        if type == "point_cloud":
            generate_output_ply(output_dict, base_url,
                                output_path, save_iterations)
        elif type == "splat_cloud":
            generate_output_splat(output_dict, base_url,
                                  output_path, save_iterations)
        elif type == "video":
            generate_output_video(output_dict, base_url,
                                  output_path, save_iterations, **kwargs)
        else:
            logger.critical(
                "Invalid output type %s given. Job: %s", type, output_dict["id"])
            raise Exception("Attempted to generate invalid output type")
