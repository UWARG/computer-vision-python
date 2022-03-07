import argparse
from datetime import datetime
import logging
from modules.targetAcquisition.taxi.TaxiWorker import taxi_worker
import os
import multiprocessing as mp
import json
from modules.targetAcquisition.targetAcquisitionWorker import targetAcquisitionWorker
from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker
from modules.decklinksrc.decklinkSrcWorker_taxi import decklinkSrcWorker_taxi
from modules.QRScanner.QRWorker import qr_worker
from modules.search.searchWorker import searchWorker
from modules.commandModule.commandWorker_flight import flight_command_worker, pogi_subworker
#from modules.commandModule.commandWorker_taxi_first import command_taxi_worker_continuous, taxi_command_worker_first
from modules.mergeImageWithTelemetry.mergeImageWithTelemetryWorker import pipelineMergeWorker
from modules.geolocation.geolocationWorker import geolocation_locator_worker, geolocation_output_worker
from modules.videoDisplay.videoDisplayWorker import videoDisplay

PIGO_DIRECTORY = ""
POGI_DIRECTORY = ""

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
    # Queue from command module out to fusion module containing timestamped telemetry data from POGI
    telemetryPipeline = mp.Queue()
    # Queue from fusion module out to targetAcquisition, containing grouped image and telemetry data from a "single time"
    mergedDataPipeline = mp.Queue()
    # Queue from targetAcquisition out to geolocation_locator_worker, containing centre-of-bbox coordinate data and associated telemetry data
    bboxAndTelemetryPipeline = mp.Lock()
    # Intermediary pipeline transferring a list of potential coordinates from geolocaion_locator_worker to geolocation_ouput_worker
    geolocationIntermediatePipeline = mp.Queue()
    # Queue from geolocation module out to command module, containing (x, y) coordinates of detected pylons
    locationCommandPipeline = mp.Queue()

    # Lock for bboxAndTelemetryPipeline
    bboxAndTelemetryLock = mp.Lock()
    # Lock for geolocationIntermediatePipeline
    geolocationIntermediateLock = mp.Lock()

    # Utility locks
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker, args=(pause, quit, videoPipeline)),
        mp.Process(target=pipelineMergeWorker,
                   args=(pause, quit, videoPipeline, telemetryPipeline, mergedDataPipeline)),
        mp.Process(target=targetAcquisitionWorker, args=(pause, quit, mergedDataPipeline, bboxAndTelemetryPipeline)),
        mp.Process(target=geolocation_locator_worker,
                   args=(pause, quit, bboxAndTelemetryPipeline, geolocationIntermediatePipeline, bboxAndTelemetryLock)),
        mp.Process(target=geolocation_output_worker, args=(
        pause, quit, geolocationIntermediatePipeline, locationCommandPipeline, geolocationIntermediateLock)),
        mp.Process(target=flight_command_worker,
                   args=(pause, quit, locationCommandPipeline, telemetryPipeline, PIGO_DIRECTORY, POGI_DIRECTORY))
    ]

    for p in processes:
        p.start()

    logger.debug("main/flightProgram: Flight program init complete")


def qrProgram():
    """
    Search program implementation here.
    Parameters: None
    Returns: None
    """
    videoPipeline = mp.Queue()

    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker_taxi, args=(pause, quit, videoPipeline)),
        mp.Process(target=qr_worker, args=(pause, quit, videoPipeline))
    ]

    for p in processes:
        p.start()

def init_logger():
    baseDir = os.path.dirname(os.path.realpath(__file__))
    logFileName = os.path.join(baseDir, "logs", str(datetime.today().date()) + "_" +
                               str(datetime.today().hour) + "." +
                               str(datetime.today().minute) + "." +
                               str(datetime.today().second) + ".log")
#    with open(logFileName, 'w') as write_file:
#        write_file.write("LOG START")

    formatter = logging.Formatter(fmt='%(asctime)s: [%(levelname)s] %(message)s', datefmt='%I:%M:%S')
    fileHandler = logging.FileHandler(filename=logFileName, mode="w")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[fileHandler, streamHandler])
    logging.debug("main/init_logger: Logger Initialized")
    return logging.getLogger()


def taxiProgram():
    """
    Taxi program implementation here.
    Parameters: None
    Returns: None
    """
    #function definition: Log 'msg % args' with severity 'ERROR'.
    logger.error("main/taxiProgram: Taxi Program Started")

    # Set up data structures for first POGI retrieval
    pogiInitPipeline = mp.Queue()
    firstTelemetry = None

    while True:
        # Read pogi data and put it into the pipeline if it is available
        pogi_subworker(pogiInitPipeline, POGI_DIRECTORY)
        
        # If we don't get any data, try again
        if pogiInitPipeline.empty():
            continue # skips the rest of the loop and loops again
        
        # Once we have data, break out of the loop
        firstTelemetry = pogiInitPipeline.get()
        break
    
    # Get cached Pylon GPS coordinates
    pylonGpsData = None
    with open("temp_pylon_gps") as file:
        pylonGpsData = json.load(file)

    # If any of the two pieces of data from above are None, throw an error and leave
    if firstTelemetry is None:
        logger.error("main/taxiProgram: Taxi program couldn't get telemetry data")
        return
    if pylonGpsData is None:
        logger.error("main/taxiProgram: Taxi program couldn't get cached pylon gps data")
        return
    
    # Get result from search and run taxi command worker with the given heading command
    searchResult = searchWorker(firstTelemetry.data, pylonGpsData)
    taxi_command_worker_first(searchResult)

    # Set up pipeline architecture for taxi
    deckLinkSrcOutTaxiInPipeline = mp.Queue() # Timestamped data
    taxiOutCommandInPipeline = mp.Queue()
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker, args=(pause, quit, deckLinkSrcOutTaxiInPipeline)),
        mp.Process(target=taxi_worker, args=(pause, quit, deckLinkSrcOutTaxiInPipeline, taxiOutCommandInPipeline)),
        mp.Process(target=command_taxi_worker_continuous, args=(pause, quit, taxiOutCommandInPipeline))
    ]

    for p in processes:
        p.start()

    logger.error("main/taxiProgram: Taxi Program Init Finished")
    return

def showVideo(): # this function needs to call functions in videoDisplay and decklinkSrcWorker
    """
    Display video implementation here.
    Parameters: None
    Returns: None
    """

#    logger.debug("main/showVideo: Video Display Started") # start message, logs with severity DEBUG

    videoPipeline = mp.Queue()
    # Queue from command module out to fusion module containing timestamped telemetry data from POGI

    # Utility locks
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker_taxi, args=(pause, quit, videoPipeline)),
        mp.Process(target=videoDisplay, args=(pause, quit, videoPipeline))
    ]

    for p in processes:
        p.start()

    #logger.debug("main/showVideo: Video Display Finished")
    return

if __name__ == '__main__': # test video in main function
    """
    Starts the appropriate program based on what was passed in as a command line argument.
    Parameters: Args for commands
    Returns: None
    """
    logger = init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="Program name to execute (flight, taxi, search)")
    # Locals is a symbol table, it allows you to execute a function by doing a search of its name.
    program = parser.parse_args().program

    assert program + 'Program' in locals()

    locals()[program + 'Program']()
