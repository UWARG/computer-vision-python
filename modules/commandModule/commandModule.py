'''

PIGO (Plane in, Ground out):

typedef struct IncomingTelemetry
{
    long double lattitudeOfObject;
    long double longitudeOfObject;

    gimbalCommands_t gimbalCommands;

    bool beginLanding;
    bool begin Takeoff;
    bool disconnectAutoPilot;

    char stuff[20];

}IncomingTelemetry_t;

GIPO (Ground in, Plane out):

typedef struct OutgoingTelemetry
{
    int errorCode;

    int currentAltitude;
    long double currentLattitude;
    long double currentLongitude;

    int sensorStatus;

    char stuff[20];

}OutgoingTelemetry_t;

- Ground station only communicates with telemetry manager, sends data after init. & gets data after report made

Question for FW:
- Update on complete struct data being processed through telemetry?
- How are instructions for the plane (pitch, roll, rudder, airspeed) being sent? Does telemetry manager send instructions to the path manager?
- Need PID controller sw for instr.?
- What's the file directory on ground station? Limits on FW side for # of files, size, location?

Question for Shrinjay:
- What instructions are needed to be sent to the plane?
- Should requirements.txt file be added for dependencies?

CommandModule Arch.
  ├── methods
  │   ### private methods ###
  │   ├── init
  │   ├── file_reader
  │   └── file_writer
  │   ### public methods ####
  │   # GIPO telemetry (getters)
  │   ├── 
  │   # PIGO telemetry (setters)
  │   └── 
      # misc.
  │   ├── get_PIGO_directory
  │   ├── set_GIPO_directory
  │   ├── get_PIGO_directory
  │   └── set_GIPO_directory
  └── properties (all private)
      # GIPO telemetry data
      ├── GIPO dict
      # PIGO telemetry data
      └── PIGO dict

dependencies needed:
- json

'''

"""
TODO:
Figure out where to put file handling: (in file readers/writers, or in set directory methods)
    - file reader/writer = tedious but integrated, set directory = need to be called first
Figure out what to do with long double data type (numpy or some other type conversion?)
    - numpy = extra dependency, type conversion = complicated
FIGURE SHIT OUT WITH FW AND SHRINJAY....SPECIFICATIONS ARE ALL OVER THE PLACE!!!!
"""

# external dependencies
#import numpy

# built in modules
import json
import logging
import sys
import os


class CommandModule:

    """
    CONSTRUCTOR
    """
    def __init__(self, gipoDirectory="", pigoDirectory=""):
        self.gipoData = dict()
        self.pigoData = dict()
        self.gipoDirectory = gipoDirectory
        self.pigoDirectory = pigoDirectory
        self.logger = logging.getLogger()

    """
    PRIVATE METHODS
    """

    """
    GET DATA FROM GIPO JSON FILE AND SAVE AS DICT
    """
    def __gipo_file_reader(self):
        try:
            with open(self.gipoDirectory, "r") as gipoFile:
                self.gipoData = json.load(gipoFile)
        except FileNotFoundError:
            self.logger.error("The given GIPO json file directory: " + self.gipoDirectory + " does not exist. Exiting...")
            sys.exit(1)
        except json.decoder.JSONDecodeError:
            self.logger.error("The given GPIO json file at: " + self.gipoDirectory + " has no data to read. Exiting...")
            sys.exit(1)

    """
    TAKE IN PIGO DICT AND WRITE TO JSON FILE
    """
    def __write_to_pigo_file(self):
        if not os.path.isfile(self.pigoDirectory):
            self.logger.warning("The given PIGO json file directory: " + self.pigoDirectory + 
                             " does not exist. Creating a new json file to populate...")
        if not bool(self.pigoData):
            self.logger.warning("The current PIGO data is empty. Writing an empty json string to: " + self.pigoDirectory + " ...")
        with open(self.pigoDirectory, "w") as pigoFile:
            json.dump(self.pigoData, pigoFile, ensure_ascii=False, indent=4, sort_keys=True)

    """
    NULL CHECKER
    """
    def __is_null(self, value):
        if value is None:
            self.logger.error("Value that was passed is null.")
            return True
        else:
            return False


    """
    PUBLIC METHODS
    """
    
    ### GIPO TELEMETRY ###
    """
    GETTERS FOR GIPO DATA
    """

    def get_error_code(self) -> int:
        self.__gipo_file_reader()
        errorCode = self.gipoData.get("errorCode")
        if type(errorCode) == int:
            return errorCode
        else:
            if errorCode == None:
                self.logger.error("Error code not found in the GIPO json file. Exiting...")
            elif type(errorCode) != int:
                self.logger.error("Error code found in the GIPO json file is not an int. Exiting...")
            sys.exit(1)

    def get_current_altitude(self) -> int:
        if self.__is_null(self.gipoData["altitude"]):
            return self.gipoData["altitude"]


    def get_current_airspeed(self) -> int:
        if self.__is_null(self.gipoData["airspeed"]):
            return self.gipoData["airspeed"]

    def get_is_landed(self) -> bool:

        if self.__is_null(self.gipoData["isLanded"]):
            return self.gipoData["isLanded"]

    def get_euler_camera(self) -> tuple:
        if self.__is_null(self.gipoData["euler_camera"]):
            euler_tuple = (self.gipoData["alpha"], self.gipoData["beta"], self.gipoData["gamma"])
            return euler_tuple

    def get_euler_plane(self) -> tuple:

        if self.__is_null(self.gipoData["euler_plane"]):
            euler_tuple = (self.gipoData["alpha"], self.gipoData["beta"], self.gipoData["gamma"])
            return euler_tuple

    def get_gps_coordinate(self) -> tuple:
        if self.__is_null(self.gipoData["euler_camera"]):
            gps_coordinate = (self.gipoData["lat"], self.gipoData["lng"], self.gipoData["alt"])
            return gps_coordinate



    ### PIGO TELEMETRY ###
    """
    SETTERS FOR PIGO DATA
    """
    def set_latitude_of_object(self, latitudeOfObject: float):

        if self.__is_null(latitudeOfObject):
            sys.exit(1)
        
        self.pigoData.update({"latitudeOfObject" : latitudeOfObject})
        self.__write_to_pigo_file()

    def set_longitude_of_object(self, longitudeOfObject):
        pass

    def set_gimbal_commands(self, gimbalCommands):
        pass

    def set_begin_landing(self, beginLanding):
        pass

    def set_begin_takeoff(self, beginTakeoff):
        pass

    def set_disconnect_autopilot(self, disconnectAutoPilot):
        pass

    ### FILE DIR METHODS ###
    def get_PIGO_directory(self):
        return self.pigoDirectory

    def get_GIPO_directory(self):
        return self.gipoDirectory

    def set_PIGO_directory(self, pigoDirectory):
        self.pigoDirectory = pigoDirectory

    def set_GIPO_directory(self, gipoDirectory):
        self.gipoDirectory = gipoDirectory

    
    # testing method used for anything
    def testing():
        pass

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    test = CommandModule("test.json", "test.json")
    test.set_latitude_of_object("test")