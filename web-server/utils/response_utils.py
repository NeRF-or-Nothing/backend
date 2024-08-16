"""
This file contains utility functions for generating responses to client requests.
"""

import gzip
import io
import json
import logging

from models.status import BaseStatus, BaseError
from flask import Response, send_file, make_response, jsonify
from typing import Optional


def make_response_metadata(
    uuid: Optional[str],
    status: BaseStatus,
    error: BaseError,
    message: Optional[str],
    **extra_data) -> dict:
    """
    Generates metadata for outward facing endpoints.
    Typically metadata will be stored in 
    'X-Metadata' header in the response.
    
    Args:
        uuid: The uuid of the resource.
        status: The status of the response.
        error: The error of the response.
        message: The message of the response.
        extra_data: Any additional data to be included in the metadata.
    """
    return {
        "uuid": uuid if uuid else "",
        "status": status.code,
        "error": error.code,
        "message": message or error.message,
        **extra_data
    }

def create_response(
    status: BaseStatus, 
    error: BaseError, 
    message: Optional[str] = None, 
    uuid: str = None,
    resource_type: str = None,
    iteration: int = None, 
    file_content: bytes = None,
    file_name: str = None,
    data: dict = None, 
    status_code: int = 200
    ) -> Response:
    """
    Returns a response object with the given status, error, message, and data.
    If file_content is provided, the response will be a file download, USE ONLY FOR
    SMALL FILES. If no additional content provided, sends empty response with header
    'X-Metadata' = {status, error, message or error.message}. 
    
    Since file sending and json sending are exclusive, can only call this
    function with either file_content or json_data.
    
    Handles the following cases:
    - File download
    - JSON data
    - Empty response
    
    Args:
        status: The status of the response.
        error: The error of the response.
        message: The message of the response.
        data: The data to be sent in the response.
        uuid: The uuid of the resource.
        resource_type: The type of the resource.
        iteration: The iteration of the resource.
        file_content: The content of the file to be sent.
        status_code: The status code of the response.
    """
    logger = logging.getLogger("web-server")
    logger.debug(f"Creating response for {uuid} {resource_type} {iteration}")
    
    extra_data = {}
    if resource_type:
        extra_data["resource_type"] = resource_type
    if iteration:
        extra_data["iteration"] = iteration
    
    metadata = make_response_metadata(uuid, status, error, message, **extra_data)
    
    if data and file_content:
        logger.error(f"Cannot send both json and file content in the same response")
        raise ValueError("Cannot send both json and file content in the same response.")
    
    response = None
    if file_content:
        response = create_response_with_file(metadata, file_content, file_name, status_code, data)
    elif data:
        response = create_response_with_json(metadata, data, status_code)
    else:
        response = make_response(json.dumps({}), status_code)
        
    response.headers['X-Metadata'] = json.dumps(metadata)
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response

def create_response_with_file(
    meta: dict, 
    file_content: bytes, 
    file_name: str,
    status_code: int,
    json_data: Optional[dict] = None
    ) -> Response:
    """
    Creates a response object with the given metadata, file content, and optional JSON data.
    
    Args:
        meta: Metadata for the response.
        file_content: The content of the file to be sent.
        file_name: The name of the file to be sent.
        status_code: The status code of the response.
        json_data: Optional JSON data to be sent along with the file.
    Raises:
        Exception: If unable to create response.
    """
    logger = logging.getLogger("web-server")
    logger.debug(f"Creating response with file content for {meta['uuid']} {file_name}")
    
    mem = io.BytesIO()
    mem.write(file_content)
    mem.seek(0)

    response = send_file(
        mem,
        as_attachment=True,
        download_name=file_name,
        mimetype="image/png",
    )

    response.headers['X-Metadata'] = json.dumps(meta)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.status_code = status_code
    
    if json_data:
        response.headers['X-JSON-Data'] = json.dumps(json_data)
    
    logger.debug(f"Response created with file content for {meta['uuid']} {file_name}")
    
    return response
    
def create_response_with_json(
    meta: dict, 
    json_data: dict, 
    status_code: int
    ) -> Response:
    """
    Creates a response object with the given metadata and file content.
    Compresses the file content before sending.
    
    Args:
        meta: Metadata for the response.
        json_data: The main data to be sent in the response. i.e Config for POST
        status_code: The status code of the response
    Raises:
        Exception: If unable to create response.
    """
    logger = logging.getLogger("web-server")
    logger.debug(f"Creating response with json content for {meta['uuid']}")
    
    response = make_response(json.dumps(json_data), status_code)
    logger.debug(f"Response created with json content for {meta['uuid']}")
    
    return response
    
    