"""
This file contains the status codes and messages that are used by the server to communicate with the client.
"""

from enum import Enum

# TODO: Restructure this. Doesnt really make sense that sfm errors are in NerfError
# TODO: Probably add ServerError, SfmError, etc ...

class NerfError(Enum):
    NO_ERROR = (0, "There is no error")
    UNKNOWN = (1, "An unknown error occurred")
    FILE_EXISTS = (2, "File already exists")
    FILE_NOT_FOUND = (3, "File not found")
    FILE_NOT_RECEIVED = (4, "File not received by server")
    VIDEO_TOO_BLURRY = (5, "Video is too blurry")
    INVALID_FILE_EXT = (6, "Invalid file extension")
    INVALID_INPUT = (7, "Invalid input provided")
    RESOURCE_UNAVAILABLE = (8, "Requested resource is unavailable")
    PROCESSING_FAILED = (9, "Processing failed")
    INTERNAL_SERVER_ERROR = (10, "Internal server error")
    INVALID_UUID = (11, "Invalid UUID")
    INVALID_RANGE = (12, "Invalid range")
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

class NerfStatus(Enum):
    PROCESSING = (0 , "Processing")
    READY = (1, "Finished Processing")
    ERROR = (2, "Error")
    
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
