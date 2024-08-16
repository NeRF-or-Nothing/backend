"""
This file contains the dataclasses for the Scene, Sfm, Nerf, Video, 
TrainingConfig, Worker, User, and QueueList representations. These dataclasses
are used to represent the data in the database and are used to serialize and
deserialize the data to and from JSON. QueueList is used to manage the list of
job ids for a certain queue.

TODO: Remove redundant extrapolations of from_dict and to_dict
"""

import copy
from typing_extensions import deprecated
import bcrypt

import numpy as np
import numpy.typing as npt

from cryptography.fernet import Fernet
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Any, Set, Tuple, TypeVar, Callable, Type, cast, Optional
from dotenv import load_dotenv


# Load environment variables from .env file at the root of the project
T = TypeVar("T")
load_dotenv()


def from_bytes(x: Any) -> bytes:
    assert isinstance(x, bytes)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_dict(x: Any) -> dict[str, Any]:
    assert isinstance(x, dict)
    return x
        

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]

def from_set(f: Callable[[Any], T], x: Any) -> Set[T]:
    assert isinstance(x, set)
    return {f(y) for y in x}

def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x

def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class NerfV2:
    """
    Finished nerf training representation
    """
    model_file_paths: Optional[Dict[int, str]] = field(default_factory=dict)
    splat_cloud_file_paths: Optional[Dict[int, str]] = field(default_factory=dict)
    point_cloud_file_paths: Optional[Dict[int, str]] = field(default_factory=dict)
    video_file_paths: Optional[Dict[int, str]] = field(default_factory=dict)
    flag: Optional[int] = 0

    @staticmethod
    def from_dict(obj: Any) -> 'NerfV2':
        """
        Converts a dictionary object to a Nerf instance.

        Args:
            obj (Any): The dictionary object to convert to a Nerf instance.
        Returns:
            Nerf: The converted Nerf instance.
        """
        assert isinstance(obj, dict)
        model_file_paths = from_union([lambda x: {int(k): str(v) for k, v in x.items()}, from_none], obj.get("model_file_paths"))
        splat_cloud_file_paths = from_union([lambda x: {int(k): str(v) for k, v in x.items()}, from_none], obj.get("splat_cloud_file_paths"))
        point_cloud_file_paths = from_union([lambda x: {int(k): str(v) for k, v in x.items()}, from_none], obj.get("point_cloud_file_paths"))
        video_file_paths = from_union([lambda x: {int(k): str(v) for k, v in x.items()}, from_none], obj.get("video_file_paths"))
        flag = from_union([from_int, from_none], obj.get("flag"))
        return NerfV2(model_file_paths, splat_cloud_file_paths, point_cloud_file_paths, video_file_paths, flag)

    def to_dict(self) -> dict:
        """
        Converts the Scene representation to a dictionary.

        Returns:
            dict: The dictionary representation of the Scene object.
        """
        result: dict = {}
        result["model_file_paths"] = from_union([lambda x: {str(k): v for k, v in x.items()}, from_none], self.model_file_paths)
        result["splat_cloud_file_paths"] = from_union([lambda x: {str(k): v for k, v in x.items()}, from_none], self.splat_cloud_file_paths)
        result["point_cloud_file_paths"] = from_union([lambda x: {str(k): v for k, v in x.items()}, from_none], self.point_cloud_file_paths)
        result["video_file_paths"] = from_union([lambda x: {str(k): v for k, v in x.items()}, from_none], self.video_file_paths)
        result["flag"] = from_union([from_int, from_none], self.flag)
        result = {k: v for k, v in result.items() if (v != None and v != {})}
        return result
    
    @staticmethod
    def empty_nerfV2() -> 'NerfV2':
        """
        Returns an empty NerfV2 object, with all fields initialized to empty values.

        Returns:
            NerfV2: The empty NerfV2 object.
        """
        nerf_v2 = NerfV2()
        nerf_v2.model_file_paths = {}
        nerf_v2.splat_cloud_file_paths = {}
        nerf_v2.point_cloud_file_paths = {}
        nerf_v2.video_file_paths = {}
        nerf_v2.flag = 0
        return nerf_v2    
    
    @classmethod
    def get_empty(cls):
        """
        Generates deep copy of static empty NerfV2
        
        Returns:
            TrainingConfig: Deepcopy of default config
        """
        return copy.deepcopy(cls.empty_nerfV2())
    
    # "Private" static immutables
    VALID_OUTPUT_TYPES: ClassVar[Dict[str, List[str]]] = {
        "gaussian": ["splat_cloud", "point_cloud", "video"],
        "tensorf": ["model", "video"]
    }

    VALID_TRAINING_MODES: ClassVar[Tuple[str, ...]] = ("gaussian", "tensorf")

    @classmethod
    def is_valid_output_type(cls, training_mode: str, output_type: str) -> bool:
        """
        Checks if the given output type is a valid output type for the NerfV2 object
        """
        return output_type in cls.VALID_OUTPUT_TYPES[training_mode]
    
    @classmethod
    def is_valid_training_mode(cls, training_mode: str) -> bool:
        """
        Checks if the given training mode is a valid training mode for the NerfV2 object
        """
        return training_mode in cls.VALID_TRAINING_MODES


@dataclass
@deprecated("Legacy Code. Old TensoRF Nerf representation")
class Nerf:
    model_file_path: Optional[str] = None
    splat_file_path: Optional[str] = None
    ply_file_path: Optional[str] = None
    rendered_video_path: Optional[str] = None
    flag: Optional[int] = 0

    @staticmethod
    def from_dict(obj: Any) -> 'Nerf':
        assert isinstance(obj, dict)
        model_file_path = from_union([from_str, from_none], obj.get("model_file_path"))
        splat_file_path = from_union([from_str, from_none], obj.get("splat_file_path"))
        ply_file_path = from_union([from_str, from_none], obj.get("ply_file_path"))
        rendered_video_path = from_union([from_str, from_none], obj.get("rendered_video_path"))
        flag = from_union([from_int, from_none], obj.get("flag"))
        return Nerf(model_file_path, splat_file_path, ply_file_path, rendered_video_path, flag)

    def to_dict(self) -> dict:
        result: dict = {}
        result["model_file_path"] = from_union([from_str, from_none], self.model_file_path)
        result["splat_file_path"] = from_union([from_str, from_none], self.splat_file_path)
        result["ply_file_path"] = from_union([from_str, from_none], self.ply_file_path)
        result["rendered_video_path"] = from_union([from_str, from_none], self.rendered_video_path)
        result["flag"] = from_union([from_int, from_none], self.flag)
        #ignore null
        result = {k:v for k,v in result.items() if v != None }
        return result

@dataclass
class Frame:
    """
    SfM Single frame representation
    """
    file_path: Optional[str] = None
    extrinsic_matrix: Optional[npt.NDArray] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Frame':
        """
        Converts a dictionary object to a Frame instance.

        Args:
            obj (Any): The dictionary object to convert to a Frame instance.
        Returns:
            Frame: The converted Frame instance.
        """
        assert isinstance(obj, dict)
        file_path = from_union([from_str, from_none], obj.get("file_path"))
        extrinsic_matrix = np.array(from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], obj.get("extrinsic_matrix")))
        return Frame(file_path, extrinsic_matrix)

    def to_dict(self) -> dict:
        """
        Converts the Frame object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the Frame object.
        """
        result: dict = {}
        result["file_path"] = from_union([from_str, from_none], self.file_path)
        result["extrinsic_matrix"] = from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], self.extrinsic_matrix.tolist())

        #ingnore null
        result = {k:v for k,v in result.items() if v}
        return result


@dataclass
class Sfm:
    """
    SfM representation of video
    """
    intrinsic_matrix: Optional[npt.NDArray] = None
    frames: Optional[List[Frame]] = None
    white_background: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Sfm':
        """
        Converts a dictionary object to a Sfm instance.

        Args:
            obj (Any): The dictionary object to convert to a Sfm instance.
        Returns:
            Sfm: The converted Sfm instance.
        """
        assert isinstance(obj, dict)
        intrinsic_matrix = np.array(from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], obj.get("intrinsic_matrix")))
        frames = from_union([lambda x: from_list(Frame.from_dict, x), from_none], obj.get("frames"))
        white_background = from_union([from_bool, from_none], obj.get("white_background"))
        return Sfm(intrinsic_matrix, frames, white_background)

    def to_dict(self) -> dict:
        """
        Converts the Sfm object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the Sfm object.
        """
        result: dict = {}
        result["intrinsic_matrix"] = from_union([lambda x: from_list(lambda x: from_list(from_float, x), x), from_none], self.intrinsic_matrix.tolist())
        result["frames"] = from_union([lambda x: from_list(lambda x: to_class(Frame, x), x), from_none], self.frames)
        result["white_background"] = from_union([from_bool, from_none], self.white_background)

        #ignore null
        result = {k:v for k,v in result.items() if v}
        return result


@dataclass
class Video:
    """
    Video representation
    """
    file_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    duration: Optional[int] = None
    frame_count: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Video':
        """
        Converts a dictionary object to a Video instance.
                
        Args:
            obj (Any): The dictionary object to convert to a Video instance.
        Returns:
            Video: The converted Video instance.
        """
        assert isinstance(obj, dict)
        file_path = from_union([from_str, from_none], obj.get("file_path"))
        width = from_union([from_int, from_none], obj.get("vid_width"))
        height = from_union([from_int, from_none], obj.get("vid_height"))
        fps = from_union([from_float, from_none], obj.get("fps"))
        duration = from_union([from_float, from_none], obj.get("duration"))
        frame_count = from_union([from_int, from_none], obj.get("frame_count"))
        return Video(file_path, width, height, fps, duration, frame_count)

    def to_dict(self) -> dict:
        """
        Converts the Video object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the Video object.
        """
        result: dict = {}
        result["file_path"] = from_union([from_str, from_none], self.file_path)
        result["width"] = from_union([from_int, from_none], self.width)
        result["height"] = from_union([from_int, from_none], self.height)
        result["fps"] = from_union([from_float, from_none], self.fps)
        result["duration"] = from_union([from_float, from_none], self.duration)
        result["frame_count"] = from_union([from_int, from_none], self.frame_count)

        #ignore null
        result = {k:v for k,v in result.items() if v}
        return result



@dataclass
class TrainingConfig:
    """
    Dataclass containing all configuration details needed for per job
    configuration for each worker.
    
    # TODO: Probably add a bunch of static functions to generate default sfm_config, nerf_config, etc
    # TODO: DONE 7/26/24 Add default and deepcopy support
    """
    sfm_config: Optional[dict[str, Any]] = None
    nerf_config: Optional[dict[str, Any]] = None

    @staticmethod
    def default_config() -> 'TrainingConfig':
        """
        Factory to generate a TrainingConfig with default worker configs

        # TODO: Replace with actual default values once testing is done
        Returns:
            TrainingConfig: Default configuration 
        """
        return TrainingConfig(
            sfm_config={
                # Add more default SfM parameters as needed
            },
            nerf_config={
                # Add more default NeRF parameters  as needed
                "training_mode" : "gaussian",
                "output_types" : ['splat_cloud'],
                "save_iterations" : [7000, 30000],
                "total_iterations" : 30000
            }
        )

    @classmethod
    def get_default(cls):
        """
        Generates deep copy of static default training config
        
        Returns:
            TrainingConfig: Deepcopy of default config
        """
        return copy.deepcopy(cls.default_config())

    @staticmethod
    def from_dict(obj: Any) -> 'TrainingConfig':
        """
        Converts a dictionary object to a TrainingConfig instance.

        Args:
            obj (Any): The dictionary object to convert to a TrainingConfig instance.
        Returns:
            TrainingConfig: The converted TrainingConfig instance.
        """
        assert isinstance(obj, dict)
        sfm_config = from_union([from_dict, from_none], obj.get("sfm_config"))
        nerf_config = from_union([from_dict, from_none], obj.get("nerf_config"))
        return TrainingConfig(sfm_config, nerf_config)

    def to_dict(self) -> dict:
        """
        Converts the TrainingConfig object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the TrainingConfig object.
        """
        result: dict = {}
        if self.sfm_config is not None:
            result["sfm_config"] = from_union([from_dict, from_none], self.sfm_config)
        if self.nerf_config is not None:
            result["nerf_config"] = from_union([from_dict, from_none], self.nerf_config)
        return result

    def update(self, new_config: dict):
        """
        Update the current TrainingConfig object with new configuration details

        Args:
            new_config (dict): The new configuration details to update the current object with
        """
        if "sfm_config" in new_config:
            self.sfm_config.update(new_config["sfm_config"])
        if "nerf_config" in new_config:
            self.nerf_config.update(new_config["nerf_config"])

@dataclass
class Scene:
    """
    Scene representation. Contains all information about a scene from all
    stages of training pipeline
    """
    id: Optional[str] = None
    status: Optional[int] = None
    video: Optional[Video] = None
    sfm: Optional[Sfm] = None
    nerf: Optional[NerfV2] = None
    config: Optional[TrainingConfig] = None 
    
    @staticmethod
    def from_dict(obj: Any) -> 'Scene':
        """
        Converts a dictionary object to a Scene instance.

        Args:
            obj (Any): The dictionary object to convert to a Scene instance.
        Returns:
            Scene: The converted Scene instance.
        """
        assert isinstance(obj, dict)
        id = from_union([from_str, from_none], obj.get("id"))
        status = from_union([from_int, from_none], obj.get("status"))
        video = from_union([Video.from_dict, from_none], obj.get("video"))
        sfm = from_union([Sfm.from_dict, from_none], obj.get("sfm"))
        nerf = from_union([NerfV2.from_dict, from_none], obj.get("nerf"))
        config = from_union([TrainingConfig.from_dict, from_none], obj.get("config"))
        return Scene(id, status, video, sfm, nerf, config)

    def to_dict(self) -> dict:
        """
        Converts the Scene object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the Scene object.
        """
        result: dict = {}
        result["id"] = from_union([from_str, from_none], self.id)
        result["status"] = from_union([from_int, from_none], self.status)
        result["video"] = from_union([lambda x: to_class(Video, x), from_none], self.video)
        result["sfm"] = from_union([lambda x: to_class(Sfm, x), from_none], self.sfm)
        result["nerf"] = from_union([lambda x: to_class(NerfV2, x), from_none], self.nerf)
        result["config"] = from_union([lambda x: to_class(TrainingConfig, x), from_none], self.config)

        #ignore null
        result = {k:v for k,v in result.items() if v}
        return result


def scene_from_dict(s: Any) -> Scene:
    """
    Non-Statically converts a dictionary object to a Scene instance. 

    Args:
        s (Any): The dictionary object to convert to a Scene instance.

    Returns:
        Scene: The converted Scene instance.
    """
    return Scene.from_dict(s)


def scene_to_dict(x: Scene) -> Any:
    """
    Non-Statically converts a Scene instance to a dictionary.

    Args:
        x (Scene): The Scene instance to convert to a dictionary.

    Returns:
        Any: The dictionary representation of the Scene instance.
    """
    return to_class(Scene, x)


# TODO: Find a use for these
@dataclass
class Worker:
    """
    Worker representation. Most likely would be used to help
    manage user access to workers when multiple of each type exist.
    For instance, free tier users should have a lot less compute limit
    than paid (ik this is FOSS just an example)
    """
    id: Optional[str] = None
    api_key: Optional[str] = None
    owner_id: Optional[str] = None
    type: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Worker':
        """
        Converts a dictionary object to a Worker instance

        Args:
            obj (Any): The dictionary object to convert to a Worker instance.
        Returns:
            Worker: The converted Worker instance.
        """
        assert isinstance(obj, dict)
        id = from_union([from_str, from_none], obj.get("_id"))
        api_key = from_union([from_str, from_none], obj.get("api_key"))
        owner_id = from_union([from_str, from_none], obj.get("owner_id"))
        type = from_union([from_str, from_none], obj.get("type"))
        return Worker(id, api_key, owner_id, type)

    def to_dict(self) -> dict:
        """
        Converts the Worker object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the Worker object.
        """
        result: dict = {}
        if self.id is not None:
            result["_id"] = from_union([from_str, from_none], self.id)
        if self.api_key is not None:
            result["api_key"] = from_union([from_str, from_none], self.api_key)
        if self.owner_id is not None:
            result["owner_id"] = from_union([from_str, from_none], self.owner_id)
        if self.type is not None:
            result["type"] = from_union([from_str, from_none], self.type)
        return result


def worker_from_dict(s: Any) -> Worker:
    """
    Non-Statically converts a dictionary object to a Worker instance.

    Args:
        s (Any): The dictionary object to convert to a Worker instance.
    Returns:
        Worker: The converted Worker instance.
    """
    return Worker.from_dict(s)


def worker_to_dict(x: Worker) -> Any:
    """
    Non-Statically converts a Worker instance to a dictionary

    Args:
        x (Worker): The Worker instance to convert to a dictionary.
    Returns:
        Any: The dictionary representation of the Worker instance.
    """
    return to_class(Worker, x)


# api_key owner
# TODO: Desperately needs better password handling. Should probably invest in salting and hashing passwords. Look into https://en.wikipedia.org/wiki/PBKDF2
# TODO: Specific password objects, or at least a password hashing function
@dataclass
class User:
    """
    User representation. Would be used to manage user access to completed scenes,
    specific workers, and other user-specific data. Stores list of scenes generated
    by user, as well as workers owned by user.
    
    TODO: Move to separate file
    TODO: Salt and hash passwords with bcrypt (1 way)
    TODO: Reimplement workers_owned functionality
    """
    # Plaintext
    username: Optional[str] = None
    _id: Optional[str] = None
    scene_ids: Set[str] = field(default_factory=set) # Sets for use, stored as list, need to convert each time
    # Encrypted
    encrypted_password: Optional[str] = None
    
    def set_password(self, password: str):
        """
        Set new password; salted and hashed with bcrypt.
        Stored as string of bytes

        Args:
            password (str): plaintext password
        """
        salt = bcrypt.gensalt()
        self.encrypted_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def check_password(self, password: str) -> bool:
        """
        Check if password is correct. Hashes and compares with stored hash
        
        Args:
            password (str): plaintext password
        Returns:
            bool: True if equal
        """
        return bcrypt.checkpw(password.encode('utf-8'), self.encrypted_password.encode('utf-8'))

    def add_scene(self, scene_id: str):
        """
        Add scene to user's scene_ids

        Args:
            scene_id (str): scene id to add
        """
        self.scene_ids.add(scene_id)

    @staticmethod
    def from_dict(obj: Any) -> 'User':
        """
        Converts a dictionary object to a User instance

        Args:
            obj (Any): The dictionary object to convert to a User instance.
        Returns:
            User: The converted User instance.
        """
        assert isinstance(obj, dict)
        username = from_union([from_str, from_none], obj.get("username"))
        _id = from_union([from_str, from_none], obj.get("_id"))
        scene_ids = set(from_union([lambda x: from_list(from_str, x), lambda _: []], obj.get("scene_ids")))
        encrypted_password = from_union([from_str, from_none], obj.get("encrypted_password"))
        return User(username, _id, scene_ids, encrypted_password)

    def to_dict(self) -> dict:
        """
        Converts the User object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the User object.
        """
        result: dict = {}
        if self.username is not None:
            result["username"] = from_union([from_str, from_none], self.username)
        if self.encrypted_password is not None:
            result["encrypted_password"] = from_union([from_str, from_none], self.encrypted_password)
        if self._id is not None:
            result["_id"] = from_union([from_str, from_none], self._id)
        if self.scene_ids:
            result["scene_ids"] = from_union([lambda x: list(x), from_none], self.scene_ids)
        # if self.workers_owned is not None:
        #     result["workers_owned"] = from_union([lambda x: from_list(from_str, x), from_none], self.workers_owned)

        return result


def user_from_dict(s: Any) -> User:
    """
    Non-Statically converts a dictionary object to a User instance.

    Args:
        s (Any): The dictionary object to convert to a User instance.

    Returns:
        User: The converted User instance.
    """
    return User.from_dict(s)


def user_to_dict(x: User) -> Any:
    """
    Non-Statically converts a User instance to a dictionary

    Args:
        x (User): The User instance to convert to a dictionary.

    Returns:
        Any: The dictionary representation of the User instance.
    """
    return to_class(User, x)

# QueueList manages list of ids
@dataclass
class QueueList:
    """
    QueueList manages a list of job ids for a certain queue. Used for reporting
    the current job progress to the user.
    """
    _id: Optional[str] = None
    queue: Optional[List[str]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'QueueList':
        """
        Converts a dictionary object to a QueueList instance

        Args:
            obj (Any): The dictionary object to convert to a QueueList instance.
        Returns:
            QueueList: The converted QueueList instance.
        """
        assert isinstance(obj, dict)
        _id = from_union([from_str, from_none], obj.get("_id"))
        queue = from_union([lambda x:from_list(from_str,x),from_none],obj.get("queue"))
        return QueueList(_id,queue)

    def to_dict(self) -> dict:
        """
        Converts the QueueList object to a dictionary. Usually for JSON serialization.

        Returns:
            dict: The dictionary representation of the QueueList object.
        """
        result: dict = {}
        if self._id is not None:
            result["_id"] = from_union([from_str, from_none], self._id)
        if self.queue is not None:
            result["queue"] = from_union([lambda x: from_list(from_str, x), from_none], self.queue)
        return result