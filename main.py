"""
This file creates the main processes for the Gaussian Splatting Worker, 
a flask server and a for the GaussianJobDispatcher.

FOR CUDA DEVICE USAGE:
Cuda is not friendly with duplicated process memory. 
flask must run in a forked process from python.multiprocessing.
training and pika must run in a spawned process from torch.multiprocessing.

If flask is not in forked process, web-server cannot send get requests,
but GaussianJobDispatcher will be able to send get requests to web-server

Additional note: Loggers are split per process to not have to do deal 
with creating a synchronization scheme. If you are interested in
unifying the logging, Python 3.2 QueueHandler and QueueListener 
(or RotatingFileHandle) can be used to do this.
Do note that since dispatcher is in its own memory space, there might be some
hiccups.

TODO: DONE 7/10/24 Split logging per proces
"""


import multiprocessing

from services.fileserver import FileServer
from services.dispatcher import GaussianJobDispatcher
from torch import multiprocessing as tmultiprocessing

def start_fileserver():
    """
    Creates file server
    """
    file_server = FileServer()
    file_server.start()

def start_dispatcher(i):
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
    ServerProcess = multiprocessing.Process(target=start_fileserver, args=())
    ServerProcess.start()
    JobProcess = tmultiprocessing.spawn(fn=start_dispatcher, args=())
    JobProcess.start()
    ServerProcess.join()
    JobProcess.join()
