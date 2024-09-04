"""
This file contains utility functions for converting between the different formats used by the gaussian splatting code and the TensoRF code.
Additioanlly, it contains a function to convert a .ply file to a .splat file for use in the frontend renderer.
"""


import logging
import json
import sys

import numpy as np

from plyfile import PlyData
from io import BytesIO
from sys import maxsize
from pathlib import Path


def convert_transforms_to_gaussian(transforms):
    '''
    Utility function to convert the transforms file from the format TensoRF input / SFM output uses to
    the format used by the gaussian splatting code. See online resources for more 
    camera/world space transformation details.

    The format we use for TensoRF is:
    ```json 
    {
        "intrinsic_matrix": [[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]],
        "vid_width": width,
        "vid_height": height,
        "frames": [
            {
                "file_path": "path/to/image",
                "extrinsic_matrix": [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3], [0, 0, 0, 1]]
            },
            ...
        ],
        ... (other metadata)
    }
    The format used by the gaussian splatting code is:
    {
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "frames": [
            {
                "file_path": "path/to/image",
                "transform_matrix": [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3], [0, 0, 0, 1]]
            },
            ...
        ],
        ... (other metadata)
    }
    ```
    '''
    logger = logging.getLogger("nerf-worker-dispatcher")
    logger.info("Converting transforms format for {transforms[\"%s\"]}", id)

    intrinsic = np.array(transforms["intrinsic_matrix"])
    width = transforms["vid_width"]
    height = transforms["vid_height"]
    fovx = 2 * np.tanh(width / (2 * intrinsic[0, 0]))
    fovy = 2 * np.tanh(height / (2 * intrinsic[1, 1]))
    transforms["camera_angle_x"] = fovx
    transforms["camera_angle_y"] = fovy

    for i, fr in enumerate(transforms["frames"]):
        fr["transform_matrix"] = fr.pop("extrinsic_matrix")

    logger.info("Finished converting transforms format for {transforms[\"%s\"]}", id)
    return transforms


def convert_transforms_to_tensorf(transforms):
    '''
    TODO: If needed
    '''
    pass


def convert_ply_to_splat(ply_file_path: Path, num_splats: int = sys.maxsize, verbose=False):
    """
    Convert a .ply file to a "compressed" .splat file for use in the
    frontend renderer. 1M splats is roughly 50MB compressed.

    Args:
        ply_file_path (Path): Path to the .ply file to convert
        num_splats (int): Number of splats to convert. Defaults to maxsize.
        verbose (bool, optional): Defaults to False.
    Returns:
        bytes: The converted splats in a byte buffer
    """
    logger = logging.getLogger("nerf-worker-dispatcher")

    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    num_converted = 0
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

        if verbose and num_converted % int(num_splats / 1000) == 0:
            logger.info("Converted %d splats", num_converted)

        num_converted += 1
        # if num_splats != 0 and num_converted == num_splats:
        #     break

    logger.info("Converted %d splats", num_converted)
    return buffer.getvalue()


if __name__ == "__main__":
    path = sys.argv[1]
    data = convert_transforms_to_gaussian(json.load(open(path, 'r')))
    with open(path, 'w') as f:
        json.dump(data, f)
