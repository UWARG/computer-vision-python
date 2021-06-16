import argparse
import os
import multiprocessing as mp
from modules.targetAcquisition.targetAcquisitionWorker import targetAcquisitionWorker
from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker
from modules.commandModule.commandWorker_flight import flight_command_worker
from modules.mergeImageWithTelemetry.mergeImageWithTelemetryWorker import pipelineMergeWorker
from modules.geolocation.geolocationWorker import geolocation_locator_worker, geolocation_output_worker

PIGO_DIRECTORY = ""

# Main process called by command line
# Main process manages PROGRAMS, programs call submodules for data processing and move data around to achieve a goal.


def callTrain():
    main_directory = os.getcwd()
    """
    stores current working directory prior to change
    """
    if os.path.exists("targetAcquisition/yolov2_assets"):
        # Importing file runs the process inside
        import modules.targetAcquisition.yolov2_assets.train
    else:
        print("YOLOV2_ASSETS Directory not found. Specify path")


def flightProgram():
    """
    Flight program implementation goes here. Outline:
        Instantiate pipeline, video mediator, start frame caputre, feed tent coordinates into pipeline.
        Feed tent coordinates from pipeline into geolocation
        Get GPS coordinates from geolocation
        Send coordinates to command module
    Parameters: None
    """
    print("start flight program")
    # Queue from decklinksrc to targetAcquisition, containing video frame data
    videoPipeline = mp.Queue()
    # Queue from targetAcquisition out to fusion module, containing pixel data about location of bbox in image
    bboxCoordinatesPipeline = mp.Queue()
    # Queue from command module out to fusion module containing telemetry data from POGI
    telemetryPipeline = mp.Queue()
    # Queue from fusion module out to geolocation locator, containing information about image pixel data & telemetry
    mergedDataPipeline = mp.Queue()
    # Lock for mergedDataPipeline
    mergedDataPipelineLock = mp.Lock()
    # Intermediary pipeline transferring data from geolocaion_locator_worker to geolocation_ouput_worker
    geolocationIntermediaryPipeline = mp.Queue()
    # Lock for geolocationIntermediaryPipeline
    geolocationIntermediaryPipelineLock = mp.Lock()
    # Queue from geolocation module out to command module, containing GPS coordinates of pylons
    coordinatePipeline = mp.Queue()

    
    # Utility locks
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker, args=(pause, quit, videoPipeline)),
        mp.Process(target=targetAcquisitionWorker, args=(pause, quit, videoPipeline, bboxCoordinatesPipeline)),
        mp.Process(target=pipelineMergeWorker, args=(pause, quit, bboxCoordinatesPipeline, telemetryPipeline, mergedDataPipeline)),
        mp.Process(target=geolocation_locator_worker, args=(pause, quit, mergedDataPipeline, geolocationIntermediaryPipeline, mergedDataPipelineLock)),
        mp.Process(target=geolocation_output_worker, args=(pause, quit, geolocationIntermediaryPipeline, coordinatePipeline, geolocationIntermediaryPipelineLock)),
        mp.Process(target=flight_command_worker, args=(pause, quit, coordinatePipeline, telemetryPipeline, PIGO_DIRECTORY))
    ]

    for p in processes:
        p.start()




def searchProgram():
    """
    Search program implementation here.
    Parameters: None
    Returns: None
    """
    return


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
    parser = argparse.ArgumentParser()
    parser.add_argument("program", help="Program name to execute (flight, taxi, search)")
    # Locals is a symbol table, it allows you to execute a function by doing a search of its name.
    program = parser.parse_args().program

    assert program + 'Program' in locals()

    locals()[program + 'Program']()
