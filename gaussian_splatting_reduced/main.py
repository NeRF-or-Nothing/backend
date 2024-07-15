"""
This file creates the main processes for the Gaussian Splatting Worker, 
a flask server and a torch multiprocessing process for the GaussianJobDispatcher.

TODO: DONE 7/10/24 Split logging per proces
TODO: Multi-GPU support
"""


import multiprocessing
import time

from services.fileserver_service import FileServer
from services.dispatch_service import GaussianJobDispatcher
from torch import multiprocessing as tmultiprocessing

def start_file_server():
    """
    Creates file server
    """
    file_server = FileServer()
    file_server.start()

def start_gaussian_job_dispatcher(i):
    """
    Creates job dispatcher and runner

    Args:
        i (_type_): torch multiprocessing index
    """
    dispatcher = GaussianJobDispatcher()
    try:
        dispatcher.start()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        dispatcher.shutdown()
                    
if __name__ == "__main__":
    
    # IMPORTANT: FOR CUDA DEVICE USAGE
    # flask must run in a normally FORKED python.multiprocessing process
    # training and pika must run in a SPAWNED torch.multiprocessing process
    # else you will have issues with redeclaring cuda devices
    # if flask is not in forked process, web-server cannot send get requests,
    # but GaussianJobDispatcher will be able to send get requests to web-server

    # additional note: Loggers are split per process to not have to do deal 
    # with creating a synchronizer for each process. If you are interested in
    # unifying the logging, Python 3.2 QueueHandler and QueueListener 
    # (or RotatingFileHandle) can be used to do this

    ServerProcess = multiprocessing.Process(target=start_file_server, args=())
    ServerProcess.start()
    JobProcess = tmultiprocessing.spawn(fn=start_gaussian_job_dispatcher, args=())
    JobProcess.start()
    ServerProcess.join()
    JobProcess.join()
