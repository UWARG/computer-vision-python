import argparse
from datetime import datetime
import logging
import os
import multiprocessing as mp
from modules.targetAcquisition.targetAcquisitionWorker import targetAcquisitionWorker
from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker

# Main process called by command line
# Main process manages PROGRAMS, programs call submodules for data processing and move data around to achieve a goal.

logger = None

def callTrain():
    main_directory = os.getcwd()
    """
    stores current working directory prior to change
    """
    logger.debug("main/callTrain: Started")

    if os.path.exists("targetAcquisition/yolov2_assets"):
        # Importing file runs the process inside
        import modules.targetAcquisition.yolov2_assets.train
    else:
        logger.error("main/callTrain: YOLOV2_ASSETS Directory not found. Specify path")
    
    logger.error("main/callTrain: Finished")


def flightProgram():
    """
    Flight program implementation goes here. Outline:
        Instantiate pipeline, video mediator, start frame caputre, feed tent coordinates into pipeline.
        Feed tent coordinates from pipeline into geolocation
        Get GPS coordinates from geolocation
        Send coordinates to command module
    Parameters: None
    """
    logger.debug("main/flightProgram: Start flight program")
    # Queue from decklinksrc to targetAcquisition
    videoPipeline = mp.Queue()
    # Queue from targetAcquisition out to main/geolocation
    coordinatePipeline = mp.Queue()
    # Utility locks
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker, args=(pause, quit, videoPipeline)),
        mp.Process(target=targetAcquisitionWorker, args=(pause, quit, videoPipeline, coordinatePipeline))
    ]

    for p in processes:
        p.start()
    
    logger.debug("main/flightProgram: Flight program init complete")




def searchProgram():
    """
    Search program implementation here.
    Parameters: None
    Returns: None
    """
    return


def init_logger():
    logFileName = os.path.join("logs", str(datetime.today().date()) + "_" +
                                       str(datetime.today().hour) + "." +
                                       str(datetime.today().minute) + "." +
                                       str(datetime.today().second) + ".log")
    
    formatter = logging.Formatter(fmt='%(asctime)s: [%(levelname)s] %(message)s', datefmt='%I:%M:%S')
    fileHandler = logging.FileHandler(filename=logFileName, mode="w")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[fileHandler, streamHandler])
    logging.debug("main/init_logger: Logger Initialized")
    logger = logging.getLogger()

def taxiProgram():
    """
    Taxi program implementation here.
    Parameters: None
    Returns: None
    """
    return


if __name__ == '__main__':
    """
    Starts the appropriate program based on what was passed in as a command line argument.
    Parameters: Args for commands
    Returns: None
    """
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="Program name to execute (flight, taxi, search)")
    # Locals is a symbol table, it allows you to execute a function by doing a search of its name.
    program = parser.parse_args().program

    assert program + 'Program' in locals()

    locals()[program + 'Program']()
