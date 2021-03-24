import argparse
# Main process called by command line
# Main process manages PROGRAMS, programs call submodules for data processing and move data around to achieve a goal.
import os
from modules.videoMediator.videoMediator import VideoMediator


def callTrain():
    main_directory = os.getcwd()

    # stores current working directory prior to change
    if os.path.exists("targetAcquisition/yolov2_assets"):
        os.chdir("targetAcquisition/yolov2_assets")

        # Changing directory to yolov2_assets to get config.json
        from modules.targetAcquisition.yolov2_assets.train import _main_
        _main_({'conf': './config.json'})
        os.chdir(main_directory)

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
    videoFeed = VideoMediator()
    # To get frame:
    frame = videoFeed.coordinates.pop(0) if len(videoFeed.coordinates) != 0 else None

    return frame


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
