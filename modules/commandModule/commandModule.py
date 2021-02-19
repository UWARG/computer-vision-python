# external dependencies
from filelock import FileLock

# built-in modules
import json
import logging
import sys
import os
from datetime import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        self.__gipo_file_reader()

class CommandModule:
    """
    Provides communication between CV system and telemetry manager to send data and instructions

    ...

    Attributes
    ----------
    pogiData : dict
        dictionary of ground-in plane-out (POGI) data to be read from the POGI json file
    pigoData : dict
        dictionary of plane-in ground-out (PIGO) data to be written to the PIGO json file
    pogiFileDirectory : str
        string containing directory to POGI file
    pigoFileDirectory : str
        string containing directory to PIGO file
    pogiLock : filelock.FileLock
        FileLock used to lock the POGI file when being read
    pigoLock : filelock.FileLock
        FileLock used to lock the PIGO file when being written
    logger : logging.Logger
        Logger for outputing log messages

    note: pogiData and pigoData are currently specified in https://docs.google.com/document/d/1j7zZJAirZ91-UeNFV-kRFMhTaT8Swr9J8PcyvD0fPpo/edit


    Methods
    -------
    PRIVATE
    __init__(pogiFileDirectory="", pigoFileDirectory="")
    __read_from_pogi_file()
    __write_to_pigo_file()
    __is_null(value)

    get_error_code() -> int
    get_current_altitude() -> float
    get_current_latitude() -> float
    get_current_longitude() -> float
    get_sensor_status() -> float

    set_gps_coordinates(gpsCoordinates: dict)
    set_ground_commands(groundCommands: dict)
    set_gimbal_commands(gimbalCommands: dict)
    set_begin_landing(beginLanding: bool)
    set_begin_takeoff(beginTakeoff: bool)
    set_disconnect_autopilot(disconnectAutoPilot: bool)

    get_POGI_directory()
    set_POGI_directory(pogiFileDirectory)
    get_PIGO_directory()
    set_PIGO_directory(pigoFileDirectory)
    """

    def __init__(self, pogiFileDirectory="", pigoFileDirectory=""):
        """
        Initializes POGI & PIGO empty dictionaries, POGI & PIGO file directories, PIGO file lock, and logger

        Parameters
        ----------
        pogiFileDirectory: str
            String of POGI file directory by default set to empty string
        pigoFileDirectory: str
            String of POGI file directory by default set to empty string
        """
        self.pogiData = dict()
        self.pigoData = dict()
        self.pogiFileDirectory = pogiFileDirectory
        self.pigoFileDirectory = pigoFileDirectory
        self.pogiLock = FileLock(pogiFileDirectory + ".lock")
        self.pigoLock = FileLock(pigoFileDirectory + ".lock")
        self.logger = logging.getLogger()
        #self.__watchdog_listener()     # temporarily disabled: producing errors

    async def __watchdog_listener(self):
        """
        Asynchronous watchdog method
        """
        if __name__ == "__main__":
            event_handler = MyHandler()
            observer = Observer()
            observer.schedule(event_handler, self.gipoDirectory, recursive=False)
            observer.start()
            try:
                while True:
                    time.sleep(1)
            finally:
                observer.stop()
                observer.join()

    def __read_from_pogi_file(self):
        """
        Decodes JSON data from pogiFileDirectory JSON file and stores it in pogiData dictionary
        """
        try:
            with self.pogiLock, open(self.pogiFileDirectory, "r") as pogiFile:
                self.pogiData = json.load(pogiFile)
        except FileNotFoundError:
            self.logger.error("The given POGI json file directory: " + self.pogiFileDirectory + " does not exist. Exiting...")
            sys.exit(1)
        except json.decoder.JSONDecodeError:
            self.logger.error("The given GPIO json file at: " + self.pogiFileDirectory + " has no data to read. Exiting...")
            sys.exit(1)

    def __write_to_pigo_file(self):
        """
        Encodes pigoData dictionary as JSON and stores it in pigoFileDirectory JSON file
        """
        if not os.path.isfile(self.pigoFileDirectory):
            self.logger.warning("The given PIGO json file directory: " + self.pigoFileDirectory + 
                             " does not exist. Creating a new json file to populate...")
        if not bool(self.pigoData):
            self.logger.warning("The current PIGO data is empty. Writing an empty json string to: " + self.pigoFileDirectory + " ...") 

        with self.pigoLock, open(self.pigoFileDirectory, "w") as pigoFile:
            json.dump(self.pigoData, pigoFile, ensure_ascii=False, indent=4, sort_keys=True)

    def __is_null(self, value) -> bool:
        """
        Logs an error if value is None and returns boolean indicating if it is None

        Parameters
        ----------
        value:
            Value to evaluate

        Returns
        -------
        bool:
            True if None, else False
        """
        if value is None:
            self.logger.error("Value that was passed is null.")
            return True
        else:
            return False

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



    def set_gps_coordinates(self, gpsCoordinates: dict):
        """
        Write GPS coordinates to PIGO JSON file

        Paramaters
        ----------
        gpsCoordinates: dict
            Dictionary containing GPS coordinates to write
        """
        if self.__is_null(gpsCoordinates):
            sys.exit(1)
        if type(gpsCoordinates) is not dict:
            self.logger.error("The given gps coordinates are " + str(type(gpsCoordinates)) + " and not a dictionary. Exiting...")
            sys.exit(1)

        self.pigoData.update({"gpsCoordinates" : gpsCoordinates})
        self.__write_to_pigo_file()

    def set_ground_commands(self, groundCommands: dict):
        """
        Write ground commands to PIGO JSON file

        Paramaters
        ----------
        groundCommands: dict
            Dictionary containing heading (float) and latestDistance (float) to write
        """
        if self.__is_null(groundCommands):
            sys.exit(1)
        if type(groundCommands) is not dict:
            self.logger.error("The given ground commands are " + str(type(groundCommands)) + " and not a dictionary. Exiting...")
            sys.exit(1)
        if "heading" not in groundCommands.keys():
            self.logger.error("The given ground command dictionary has no 'heading' key. Exiting...")            
        if "latestDistance" not in groundCommands.keys():
            self.logger.error("The given ground command dictionary has no 'latestDistance' key. Exiting...")
            sys.exit(1)
        if groundCommands["heading"] is not float:
            self.logger.error("The heading in the ground command dictionary is not a float. Exiting...")
            sys.exit(1)
        if groundCommands["latestDistance"] is not float:
            self.logger.error("The latestDistance in the ground command dictionary is not a float. Exiting...")
            sys.exit(1)

        self.pigoData.update({"gimbalCommands" : gimbalCommands})
        self.__write_to_pigo_file()

    def set_gimbal_commands(self, gimbalCommands: dict):
        """
        Write gimbal commands to PIGO JSON file

        Paramaters
        ----------
        gimbalCommands: dict
            Dictionary containing gimbal commands to write
        """
        if self.__is_null(gimbalCommands):
            sys.exit(1)
        if type(gimbalCommands) is not dict:
            self.logger.error("The given gimbal commands is " + str(type(gimbalCommands)) + " and not a dictionary. Exiting...")
            sys.exit(1)

        self.pigoData.update({"gimbalCommands" : gimbalCommands})
        self.__write_to_pigo_file()

    def set_begin_landing(self, beginLanding: bool):
        """
        Write begin landing command to PIGO JSON file

        Paramaters
        ----------
        beginLanding: bool
            Boolean containing whether or not to initiate landing sequence
        """
        if self.__is_null(beginLanding):
            sys.exit(1)
        if type(beginLanding) is not bool:
            self.logger.error("The given begin landing is " + str(type(beginLanding)) + " and not a boolean. Exiting...")
            sys.exit(1)

        self.pigoData.update({"beginLanding" : beginLanding})
        self.__write_to_pigo_file()

    def set_begin_takeoff(self, beginTakeoff: bool):
        """
        Write begin takeoff command to PIGO JSON file

        Paramaters
        ----------
        beginTakeoff: bool
            Boolean containing whether or not to initiate takeoff sequence
        """
        if self.__is_null(beginTakeoff):
            sys.exit(1)
        if type(beginTakeoff) is not bool:
            self.logger.error("The given begin takeoff is " + str(type(beginTakeoff)) + " and not a boolean. Exiting...")
            sys.exit(1)

        self.pigoData.update({"beginTakeoff" : beginTakeoff})
        self.__write_to_pigo_file()

    def set_disconnect_autopilot(self, disconnectAutoPilot: bool):
        """
        Write disconnect autopilot to PIGO JSON file

        Paramaters
        ----------
        disconnectAutoPilot: bool
            Boolean containing whether or not to disconnect auto pilot
        """
        if self.__is_null(disconnectAutoPilot):
            sys.exit(1)
        if type(disconnectAutoPilot) is not bool:
            self.logger.error("The given disconnect auto pilot is " + str(type(disconnectAutoPilot)) + " and not a boolean. Exiting...")
            sys.exit(1)

        self.pigoData.update({"disconnectAutoPilot" : disconnectAutoPilot})
        self.__write_to_pigo_file()

    def get_PIGO_directory(self):
        """
        Return PIGO JSON file directory

        Returns
        ----------
        pigoFileDirectory: str
            Contains directory to PIGO JSON file
        """
        return self.pigoFileDirectory

    def get_POGI_directory(self):
        """
        Return POGI JSON file directory

        Returns
        ----------
        pogiFileDirectory: str
            Contains directory to POGI JSON file
        """
        return self.pogiFileDirectory

    def set_PIGO_directory(self, pigoFileDirectory: str):
        """
        Sets PIGO JSON file directory

        Parameters
        ----------
        pigoFileDirectory: str
            Contains directory to PIGO JSON file
        """
        self.pigoFileDirectory = pigoFileDirectory

    def set_POGI_directory(self, pogiFileDirectory: str):
        """
        Sets POGI JSON file directory

        Parameters
        ----------
        pogiFileDirectory: str
            Contains directory to POGI JSON file
        """
        self.pogiFileDirectory = pogiFileDirectory