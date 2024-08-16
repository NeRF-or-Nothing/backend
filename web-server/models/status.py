"""
This file contains the status codes and messages that are used by the server to communicate with the client.

TODO: IMPORTANT - This should really be moved to external module that all workers can pull from, else risk multiple conflicting definitions
TODO: Restructure this. Doesnt really make sense that sfm errors are in NerfError. Or do traditional error code reservation (i.e nerf is 0-2000, etc.)
TODO: Probably add ServerError, SfmError, UserError, etc ...
"""

class BaseError:
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

    def __eq__(self, other):
        if isinstance(other, BaseError):
            return self.code == other.code
        return False

class BaseStatus:
    SUCCESS = BaseError(0, "Success")
    ERROR = BaseError(1, "Error")

class NerfError:
    NO_ERROR = BaseError(0, "There is no error")
    UNKNOWN = BaseError(1, "An unknown error occurred")
    FILE_EXISTS = BaseError(2, "File already exists")
    FILE_NOT_FOUND = BaseError(3, "File not found")
    FILE_NOT_RECEIVED = BaseError(4, "File not received by server")
    VIDEO_TOO_BLURRY = BaseError(5, "Video is too blurry")
    INVALID_FILE_EXT = BaseError(6, "Invalid file extension")
    INVALID_INPUT = BaseError(7, "Invalid input provided")
    RESOURCE_UNAVAILABLE = BaseError(8, "Requested resource is unavailable")
    PROCESSING_FAILED = BaseError(9, "Processing failed")
    INTERNAL_SERVER_ERROR = BaseError(10, "Internal server error")
    INVALID_UUID = BaseError(11, "Invalid UUID")
    INVALID_RANGE = BaseError(12, "Invalid range")

class NerfStatus(BaseStatus):
    PROCESSING = BaseError(2, "Processing")
    READY = BaseError(3, "Finished Processing")

class UserError:
    NO_ERROR = BaseError(0, "There is no error")
    USER_NOT_FOUND = BaseError(1, "User not found")
    INCORRECT_PASSWORD = BaseError(2, "Incorrect password")
    INVALID_JWT = BaseError(3, "Invalid JWT")
    EMAIL_ALREADY_EXISTS = BaseError(4, "Email already exists")
    USERNAME_ALREADY_EXISTS = BaseError(5, "Username already exists")
    ID_ALREADY_EXISTS = BaseError(6, "ID already exists")
    INVALID_EMAIL = BaseError(7, "Invalid email")
    INVALID_USERNAME = BaseError(8, "Invalid username")
    INVALID_ID = BaseError(9, "Invalid ID")
    INVALID_PASSWORD = BaseError(10, "Invalid password")
    UNAUTHORIZED = BaseError(11, "Unauthorized Access")

class UserStatus(BaseStatus):
    pass  # It already inherits SUCCESS and ERROR from BaseStatus

class SceneError(BaseError):
    SCENE_NOT_FOUND = BaseError(1, "Scene not found")
    SCENE_CONFLICT = BaseError(2, "Scene conflict")
    SCENE_TRAINING_FAILED = BaseError(3, "Scene training failed")