"""
This file contains output generation utilities for gaussian splatting and overall vidtonerf api formatting

TODO: Maybe remove kwargs stuff. More confusing than not
"""

import os
import render
from utils.nerf_utils import convert_ply_to_splat

from typing import List, Optional
from pathlib import Path


def generate_output_ply(base_url: str, output_path: Path, job_id: str, save_iterations: List[int]) -> dict:
    """
    Handles .ply output file generation. .ply files are 
    already generated for each snapshot interval given by
    inria train.py.
    Args:
        base_url (str): root url for worker
        output_path (Path): relative path to this jobs output directory
        job_id (str): unique identifier for this job
        save_iterations (list[int]): list of snapshot iterations to save
    Returns: 
        dict: endpoints to probe to retrieve .ply files
    """
    endpoints = {}
    for iteration in save_iterations:
        endpoints[iteration] = f"{base_url}{output_path}/point_cloud/iteration_{iteration}/{job_id}.ply"
    
    return endpoints

def generate_output_splat(base_url: str, output_path: Path, job_id: str, save_iterations: List[int]) -> dict:
    """
    Handles generation of compressed .splat file from .ply. 

    Args:
        base_url (str): root url for worker
        output_path (Path): relative path to this jobs output directory
        job_id (str): unique identifier for this job
        save_iterations (list[int]): list of snapshot iterations to save
    Returns:
        dict: endpoints to probe to retrieve .splat files
    """
    endpoints = {}
    for iteration in save_iterations:
        ply_path = output_path / f"point_cloud/iteration_{iteration}/{job_id}.ply"
        splat_bytes = convert_ply_to_splat(ply_path, 10_000_000)
        
        splat_dir = output_path / f"splat_cloud/iteration_{iteration}"
        os.makedirs(splat_dir, exist_ok=True)
        splat_path = splat_dir / f"{job_id}.splat"
       
        open(splat_path, "wb").write(splat_bytes)
        endpoints[iteration] = f"{base_url}{splat_path}" # Another intentional missing '/'
    
    return endpoints

def generate_output_video(base_url: str, output_path: Path, job_id, save_iterations: list, resolution_width: Optional[int]) -> dict:
    """
    Handles generation of video following cameras as given by 
    output_path/cameras.json.
    
    TODO: Implement video generation and this function

    Args:
        base_url (Path): root url for worker
        output_path (Path): relative path to this jobs output directory
        save_iterations (list[int]): list of snapshot iterations to save
        resolution_width (int | None): width of video outputs
    Raises:
        Exception: If no iterations are provided to save video outputs
    """
    raise NotImplementedError("generate_output_video is not implemented")

def populate_outputs(base_url: str, output_path: Path, job_id: str, output_types: List[str], save_iterations: List[int], resolution_width: Optional[int]) -> dict:
    """
    Handles generation of each unique file type that can be 
    created by gaussian splatting. Populates a output dictionary to contain
    appropriate GET endpoints to retrieve outputs
    
    Args:
        base_url (str): root url for worker
        output_path (Path): relative path to this jobs output directory
        job_id (str): unique identifier for this job
        output_types (str): types of output files to generate
        save_iterations (list[int]): list of saved snapshot iterations for which to generate outputs
        resolution_width (int | None): width of video outputs
    Returns: 
        dict: dictionary of output types to their respective endpoints
    """
    file_paths = {}
    for type in output_types:
        file_paths[type] = {}

        if type == "point_cloud":
            file_paths[type] = generate_output_ply(base_url,output_path, job_id, save_iterations)
        elif type == "splat_cloud":
            file_paths[type] = generate_output_splat(base_url, output_path, job_id, save_iterations)
        elif type == "video":
            file_paths[type] = generate_output_video(base_url, output_path, job_id, save_iterations, resolution_width)
        else:
            raise Exception(f"Attempted to generate invalid output type, Job {job_id}")

    return file_paths