from modules.commandModule.commandModule import CommandModule

PIGO_DIR = ""
POGI_DIR = ""

def write_pigo(newPigo):
#Boolean to check if any changes were made
    isChanged = False
    command = CommandModule(pigoFileDirectory=PIGO_DIR)

# temporary PIGO dictionary that calls various functions to update the values in the passed-in dictionary
    tempPigo = {
    'errorCode' : command.get_error_code(),
    'gpsCoordinates' : command.get_gps_coordinates(),
    'currentAltitude' : command.get_current_altitude(),
    'currentAirspeed' : command.get_current_airspeed(),
    'eulerAnglesZXY' : command.get_euler_angles_of_plane(),
    'eulerAnglesZXY' : command.get_euler_angles_of_camera(),
    'isLanded' : command.get_is_landed()
  }
 
# checks for any new values. If there is then isChanged is set to true
    for tempValue, pigoValue in zip(tempPigo.values(), newPigo.values()):
        if tempValue != pigoValue:
            isChanged = True
    if len(tempPigo.keys())!= len(newPigo.keys()):
        isChanged = True
#checks if the length of the two dictionaries are different, indicating a variable was added
    if isChanged :
        command.__pigoData = newPigo
	command.__write_to_pigo_file()
	return
#if nothing has changed, return the dictionary that was passed in
    else:
        return

def read_pogi() -> dict:

    """
    Reads all data currently in the POGI file and returns as a dict

    Returns
    -------
    dict:
        contains pogiData as specified in commandModule
    """

    command = CommandModule(pigoFileDirectory=PIGO_DIR, pogiFileDirectory=POGI_DIR)

    # store all current POGI data into a dict
    # note: the keys included in the dict are those which are currently valid and working in commandModule
    #       (some fields are defined in the wiki but are currently commented out)
    pogiData =  { 'currentAltitude' : command.get_current_altitude(),
                  'currentAirspeed' : command.get_current_airspeed(),
                  'isLanded' : command.get_is_landed(),
                  'eulerAnglesOfCamera' : command.get_euler_angles_of_camera(),
                  'eulerAnglesOfPlane' : command.get_euler_angles_of_plane(),
                  'gpsCoordinates' : command.get_gps_coordinates(),
                }
    
    return pogiData