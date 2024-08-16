"""
This file contains the managers for the database. The managers are used to interact with the 
database and perform CRUD operations on the data. Moved the managers to a separate file to
improve code organization and readability.
"""

import os

from models.scene import NerfV2, Nerf, Video, Sfm, TrainingConfig, QueueList
from models.scene import Scene, scene_from_dict, scene_to_dict 
from models.scene import User, user_from_dict, user_to_dict
from models.scene import worker_from_dict, worker_to_dict
from models.status import UserStatus, UserError
from typing import Optional, Tuple
from typing_extensions import deprecated
from uuid import uuid4
from pymongo import MongoClient
from cryptography.fernet import Fernet


class QueueListManager:
    """
    keeps track of lists (queues) in parallel with RabbitMQ to report queue status
    VALID QUEUE IDS: sfm_list nerf_list queue_list (queue_list is the overarching process)
    """
    
    def __init__(self, unittest=False) -> None:
        """
        Creates a new QueueListManager object.

        Args:
            unittest (bool, optional): Use for running tests. Defaults to False.
        """
        # unittest=True implies this runs on localhost for unit testing
        mongoip = "localhost" if unittest else str(os.getenv("MONGO_IP"))
        client = MongoClient(host=mongoip,port=27017,username=str(os.getenv("MONGO_INITDB_ROOT_USERNAME")),\
                             password=str(os.getenv("MONGO_INITDB_ROOT_PASSWORD")))
        self.db = client["nerfdb"]
        self.collection = self.db["queues"]
        self.upsert=True
        # Valid queue ids:
        self.queue_names = ["sfm_list","nerf_list","queue_list"]

    def __set_queue(self, _id: str, queue_list: QueueList):
        """
        Sets a queue list in the database.

        Args:
            _id (str): id for the queue list
            queue_list (QueueList): queue list object
        Raises:
            Exception: If the queue id is not valid
        """
        if _id not in self.queue_names:
            raise Exception("Not a valid queue ID. Valid queue IDs: {}".format(self.queue_names))
        key = {"_id":_id}
        value = {"$set": queue_list.to_dict()}
        self.collection.update_one(key, value, upsert=self.upsert)

    def append_queue(self, queueid: str, uuid: str):
        """
        Appends a uuid to a queue list.

        Args:
            queueid (str): id of queue to append to
            uuid (str): uuid to append
        Raises:
            Exception: If invalid queue id or uuid is already in the queue
        """
        # Check for valid queue id
        if queueid not in self.queue_names:
            raise Exception("Not a valid queue ID. Valid queue IDs: {}".format(self.queue_names))
        # Create queue or add to existing queue
        doc = self.collection.find_one(queueid)
        if not doc:
            self.__set_queue(queueid,QueueList(queueid,[uuid]))
        else:
            queue_list = QueueList.from_dict(doc)
            # Make sure ID is not in list already
            x = [x for x in queue_list.queue if x == uuid]
            if len(x) > 0:
                raise Exception("ID is already in the queue!")
            # Append queue
            queue_list.queue.append(uuid)
            self.__set_queue(queueid,queue_list)

    def get_queue_position(self, queueid: str, uuid: str) -> 'tuple[int,int]':
        """
        Returns the position of a uuid in a queue.

        Args:
            queueid (str): queue to probe
            uuid (str): uuid to find
        Raises:
            Exception: If invalid queue id or uuid is not in the queue
        Returns:
            tuple[int,int]: (position, length of queue)
        """
        # Check for valid queue id
        if queueid not in self.queue_names:
            raise Exception("Not a valid queue ID. Valid queue IDs: {}".format(self.queue_names))
        doc = self.collection.find_one(queueid)
        queue_list = QueueList.from_dict(doc)
        # Obtain indices of all occurences of the uuid
        x = [x for x in range(0,len(queue_list.queue)) if queue_list.queue[x] == uuid]
        if len(x) > 1:
            raise Exception("Same ID found multiple times in queue!")
        elif len(x) == 0:
            raise Exception("ID not found in queue!")
        else:
            return (x[0], len(queue_list.queue))
    
    def get_queue_size(self, queueid: str) -> int:
        """
        Returns the size of a queue.

        Args:
            queueid (str): queue to probe
        Raises:
            Exception: If invalid queue id
        Returns:
            int: length of queue
        """
        if queueid not in self.queue_names:
            raise Exception("Not a valid queue ID. Valid queue IDs: {}".format(self.queue_names))
        doc = self.collection.find_one(queueid)
        queue_list = QueueList.from_dict(doc)
        return len(queue_list.queue)
    
    def pop_queue(self, queueid: str, uuid: str=None):
        """
        Pops a uuid from the queue.

        Args:
            queueid (str): queue to pop from
            uuid (str, optional): uuid to pop. Defaults to None.
        Raises:
            Exception: If invalid queue id or queue empty or uuid not in queue
        """
        # Check if valid queue id
        if queueid not in self.queue_names:
            raise Exception("Not a valid queue ID. Valid queue IDs: {}".format(self.queue_names))
        doc = self.collection.find_one(queueid)
        queue_list = QueueList.from_dict(doc)
        # Check that the uuid exists or no uuid was provided
        if len(queue_list.queue) == 0 or (uuid and uuid not in queue_list.queue):
            raise Exception("Queue empty or ID not found!")
        if uuid:
            queue_list.queue.remove(uuid)
        else:
            queue_list.queue.pop(0)
        self.__set_queue(queueid, queue_list)


class SceneManager:
    """
    Manages storage and retrieval of scenes in the database
    
    TODO: Need to decide if should continue to return None, Raise Exception, or return a default object
    TODO: define set update get and delete for each object adds scene to the collection replacing any existing scene with the same id
    TODO: DONE 7/11/24 add worker configs per job, so that dynamic training/output modes can be supports
    """
    def __init__(self) -> None:
        client = MongoClient(host=str(os.getenv("MONGO_IP")),port=27017,username=str(os.getenv("MONGO_INITDB_ROOT_USERNAME")),\
                             password=str(os.getenv("MONGO_INITDB_ROOT_PASSWORD")))
        self.db = client["nerfdb"]
        self.collection = self.db["scenes"]
        self.upsert=True
    
    
    def set_training_config(self, _id: str, config: TrainingConfig):
        """
        Sets the training configuration for a scene.

        Args:
            _id (str): scene id
            config (TrainingConfig): training configuration
        """
        key = {"_id": _id}
        fields = {"config."+k:v for k,v in config.to_dict().items()}
        value = {"$set": fields}
        self.collection.update_one(key, value, upsert=self.upsert)
    
    def set_scene(self, _id: str, scene: Scene):
        """
        Sets a scene in the database
        
        Args:
            _id (str): scene id
            scene (Scene): scene object
        """
        key = {"_id": _id}
        value = {"$set": scene.to_dict()}
        self.collection.update_one(key, value, upsert=self.upsert)

    def set_video(self, _id: str, vid: Video):
        """
        Sets the video object for a scene.
        
        Args:
            _id (str): scene id
            vid (Video): video object
        """
        key = {"_id":_id}
        fields = {"video."+k:v for k,v in vid.to_dict().items()}
        value = {"$set": fields}
        self.collection.update_one(key, value, upsert=self.upsert)

    def set_sfm(self, _id: str, sfm: Sfm):
        """
        Sets the sfm object for a scene.
        
        Args:
            _id (str): scene id
            sfn (Sfm): sfm object
        """
        key = {"_id":_id}
        fields = {"sfm."+k:v for k,v in sfm.to_dict().items()}
        value = {"$set": fields}
        self.collection.update_one(key, value, upsert=self.upsert)

    def set_nerfV2(self, _id: str, nerf: NerfV2):
        """
        Sets the nerf object for a scene

        Args:
            _id (str): scene id
            nerf (NerfV2): nerfv2 object
        """
        key = {"_id":_id}
        fields = {"nerf."+k:v for k,v in nerf.to_dict().items()}
        value = {"$set": fields}
        self.collection.update_one(key, value, upsert=self.upsert)

    def set_scene_name(self, _id: str, name: str):
        """
        Sets the name of a scene.

        Args:
            _id (str): scene id
        """
        key = {"_id":_id}
        value = {"$set": {"name":name or ""}}
        self.collection.update_one(key, value, upsert=self.upsert)        
    
    @deprecated("Legcay Code. Use set_nerfV2 instead")
    def set_nerf(self, _id: str, nerf: Nerf):
        """
        Sets the nerf object for a scene

        Args:
            _id (str): scene id
            nerf (Nerf): nerf object
        """
        key = {"_id":_id}
        fields = {"nerf."+k:v for k,v in nerf.to_dict().items()}
        value = {"$set": fields}
        self.collection.update_one(key, value, upsert=self.upsert)

    def get_scene_name(self, _id: str) -> Optional[str]:
        """
        Gets the name of a scene.

        Args:
            _id (str): scene id
        Returns:
            Optional[str]: name of the scene
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc:
            return doc.get("name")
        else:
            return None

    def get_training_config(self, _id: str) -> Optional[TrainingConfig]:
        """
        Gets the training configuration for a scene.

        Args:
            _id (str): scene id
        Returns:
            Optional[TrainingConfig]: training configuration
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc: 
            return TrainingConfig.from_dict(doc["config"])
        else:
            return None

    def get_scene(self, _id: str) -> Optional[Scene]:
        """
        Gets a scene from the database.

        Args:
            _id (str): scene id
        Returns:
            Optional[Scene]: scene object
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc:
            return scene_from_dict(doc)
        else:
            return None

    def get_video(self, _id: str) -> Optional[Video]:
        """
        Gets the video object for a scene.

        Args:
            _id (str): scene id
        Returns:
            Optional[Video]: video object
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc and "video" in doc:
            return Video.from_dict(doc["video"])
        else:
            return None

    def get_sfm(self, _id: str) -> Optional[Sfm]:
        """
        Gets the sfm object for a scene.

        Args:
            _id (str): scene id
        Returns:
            Optional[Sfm]: sfm object
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc and "sfm" in doc:
            return Sfm.from_dict(doc["sfm"])
        else:
            return None
        
    @deprecated("Legacy Code. Use get_nerfV2 instead")
    def get_nerf(self, _id: str) -> Optional[Nerf]:
        """
        Gets the nerf object for a scene.

        Args:
            _id (str): scene id

        Returns:
            Optional[Nerf]: nerf object
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc and "nerf" in doc:
            return Nerf.from_dict(doc["nerf"])
        else:
            return None

    def get_nerfV2(self, _id: str) -> Optional[NerfV2]:
        """
        Gets the nerf object for a scene.

        Args:
            _id (str): scene id
        Returns:
            Optional[NerfV2]: nerf object
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc and "nerf" in doc:
            return NerfV2.from_dict(doc["nerf"])
        else:
            return None

class UserManager:
    """
    Manages storage and retrieval of users in the database. 
    user ID & Username must be unique. Uses encryption key for API key encryption.
    Will generate new encryption key if none is provided.
    
    TODO: Probably switch over to UserStatus, UserError
    """
    
    def __init__(self, unittest=False) -> None:
        # unittest=True implies this runs on localhost for unit testing
        mongoip = "localhost" if unittest else str(os.getenv("MONGO_IP"))
        client = MongoClient(host=mongoip,port=27017,username=str(os.getenv("MONGO_INITDB_ROOT_USERNAME")),\
                             password=str(os.getenv("MONGO_INITDB_ROOT_PASSWORD")))
        self.db = client["nerfdb"]
        self.collection = self.db["users"]
        self.upsert=True

    def set_user(self, user: User) -> Tuple[UserStatus, UserError]:
        """
        Sets a user in the database. If the user already exists returns an error.
        Assumes User object contains valid already encrypted pw/apikey
        
        TODO: DONE 7/15/24 This should be cleaned up.
        TODO: Really should allow user infomration to be updated
        
        Args:
            user (User): user object
        Returns:
            UserStatus, UserError
        """
        key={"username":user.username}
        doc = self.collection.find_one(key)
        if doc!=None:
            return UserStatus.ERROR, UserError.USERNAME_ALREADY_EXISTS
        
        key={"_id":user._id}
        doc = self.collection.find_one(key)
        if doc!=None:
            return UserStatus.ERROR, UserError.ID_ALREADY_EXISTS

        value = {"$set": user.to_dict()}
        self.collection.update_one(key,value,upsert=self.upsert)
        return UserStatus.SUCCESS, UserError.NO_ERROR

    def update_user(self, user: User) -> Tuple[UserStatus, UserError]:
        """
        Updates a user in the database.

        Args:
            user (User): user object to update
        Returns:
            Tuple[UserStatus, UserError]: Status of the operation
        """
        key = {"_id": user._id}
        value = {"$set": user.to_dict()}
        result = self.collection.update_one(key, value, upsert=self.upsert)
        
        if result.modified_count > 0 or result.upserted_id:
            return UserStatus.SUCCESS, UserError.NO_ERROR
        else:
            return UserStatus.ERROR, UserError.USER_NOT_FOUND
        
    def generate_user(self, username:str, password:str) -> Tuple[UserStatus, UserError]:
        """
        Generates a new user object and sets it in the database.
        
        Args:
            username (str): Username
            password (str): Password
        Returns:
            Tuple[UserStatus, UserError]: Status of the operation
        """
        _id = str(uuid4())
        
        # Check if generated user id already exists
        while self.get_user_by_id(_id) != UserError.USER_NOT_FOUND:
            _id = str(uuid4())
        
        user = User(username, _id)
        user.set_password(password)
        return self.set_user(user)
        
    def get_user_by_id(self, _id: str) -> User | UserError:
        """
        Gets a user by ID.

        Args:
            _id (str): user id
        Returns:
            User | UserError : User object if found, Error if not
        """
        key = {"_id":_id}
        doc = self.collection.find_one(key)
        if doc:
            return User.from_dict(doc)
        else:
            return UserError.USER_NOT_FOUND

    def get_user_by_username(self, username: str) -> User | UserError:
        """
        Gets a user by username.

        Args:
            username (str): username
        Returns:
            User | UserError : User object if found, Error if not
        """
        key = {"username":username}
        doc = self.collection.find_one(key)
        if doc:
            return User.from_dict(doc)
        else:
            return UserError.USER_NOT_FOUND

    def user_has_job_access(self, user_id: str, job_id: str) -> bool:
        """
        Validates that the user has access to the job.

        Args:
            user_id (str): user id
            job_id (str): job id
        Returns:
            bool: True if user has access, False if not
        """
        return job_id in self.get_user_by_id(user_id).scene_ids