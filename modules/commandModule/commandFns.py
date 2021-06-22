import json
import os
from modules.commandModule.commandModule import CommandModule

PIGO_DIR = ""
POGI_DIR = ""


def json_changed(latestJsonDirectory, currentDict) -> bool:
    """
    Compares json values to a dict

    Parameters
    ----------
    latestJsonDirectory: str
        filepath to the latest json file to compare to
    currentDict: dict
        dictionary containing values to compare with

    Returns
    -------
    bool:
        True if json file and the dict do not match, False otherwise
    """

    # if latest pogi data file doesn't exist return true
    if not os.path.isfile(latestJsonDirectory):
        return True

    # otherwise, get the latest json dict and compare
    with open(latestJsonDirectory, 'r') as f:
        latestDict = json.load(f)

    # if current and latest pogi data doesnt match size, then something changed
    if len(latestDict.keys()) != len(currentDict.keys()):
        return True

    # if keys in current pogi don't exist in latest pogi or if the values for the keys
    # don't match, then something changed
    for key in currentDict:
        if not (key in latestDict and latestDict[key] == currentDict[key]):
            return True

    return False


def write_pigo(newPigo):
    # Boolean to check if any changes were made
    isChanged = False
    command = CommandModule(pigoFileDirectory=PIGO_DIR)

    # temporary PIGO dictionary that calls various functions to update the values in the passed-in dictionary
    tempPigo = {
        'errorCode': command.get_error_code(),
        'gpsCoordinates': command.get_gps_coordinates(),
        'currentAltitude': command.get_current_altitude(),
        'currentAirspeed': command.get_current_airspeed(),
        'eulerAnglesZXY': command.get_euler_angles_of_plane(),
        'eulerAnglesZXY': command.get_euler_angles_of_camera(),
        'isLanded': command.get_is_landed()
    }

    # checks for any new values. If there is then isChanged is set to true
    for tempValue, pigoValue in zip(tempPigo.values(), newPigo.values()):
        if tempValue != pigoValue:
            isChanged = True
    if len(tempPigo.keys()) != len(newPigo.keys()):
        isChanged = True
    # checks if the length of the two dictionaries are different, indicating a variable was added
    if isChanged:
        command.__pigoData = newPigo
        command.__write_to_pigo_file()
        return
    # if nothing has changed, return the dictionary that was passed in
    else:
        return


def read_pogi(POGI_DIR="") -> tuple:
    """
    Reads all data currently in the POGI file and returns as a dict

    Parameters
    ----------
    POGI_DIR: str
        directory to pogi json file; default == ""

    Returns
    -------
    tuple:
        contains changed flag and pogi dict; i.e. (true, pogi) --> pogi is different from latest_pogi.json
    """

    command = CommandModule(pigoFileDirectory=PIGO_DIR, pogiFileDirectory=POGI_DIR)

    # store all current POGI data into a dict
    # note: the keys included in the dict are those which are currently valid and working in commandModule
    #       (some fields are defined in the wiki but are currently commented out)
    pogiData = {
        'currentAltitude': command.get_current_altitude(),
        'currentAirspeed': command.get_current_airspeed(),
        'isLanded': command.get_is_landed(),
        'eulerAnglesOfCamera': command.get_euler_angles_of_camera(),
        'eulerAnglesOfPlane': command.get_euler_angles_of_plane(),
        'gpsCoordinates': command.get_gps_coordinates(),
    }

    latestJsonDirectory = os.path.join(os.getcwd(), "modules", "commandModule", "latest_pogi.json")
    isChanged = json_changed(latestJsonDirectory, pogiData)

    # if pogi has changed, then write new data to latest_pogi.json
    if isChanged:
        with open(latestJsonDirectory, 'w') as pogiFile:
            json.dump(pogiData, pogiFile, ensure_ascii=False, indent=4, sort_keys=True)

    return isChanged, pogiData
