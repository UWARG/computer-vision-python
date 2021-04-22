# external dependencies
from filelock import FileLock

# built-in modules
import json
import logging
import sys
import os
import time
import numpy as np

class CommandModule:
    """
    Provides communication between CV and FW to send data and instructions via JSON files

    ...

    Attributes
    ----------
    __pogiData : dict
        dictionary of plane-out ground-in (POGI) data to be read from the POGI json file
    __pigoData : dict
        dictionary of plane-in ground-out (PIGO) data to be written to the PIGO json file
    pogiFileDirectory : str
        string containing directory to POGI file
    pigoFileDirectory : str
        string containing directory to PIGO file
    __pogiLock : filelock.FileLock
        FileLock used to lock the POGI file when being read
    __pigoLock : filelock.FileLock
        FileLock used to lock the PIGO file when being written
    __logger : logging.Logger
        Logger for outputing log messages

    note: pogiData and pigoData are currently specified in https://docs.google.com/document/d/1j7zZJAirZ91-UeNFV-kRFMhTaT8Swr9J8PcyvD0fPpo/edit


    Methods
    -------
    __init__(pogiFileDirectory="", pigoFileDirectory="")
    __read_from_pogi_file()
    __write_to_pigo_file()
    __is_null(value)

    get_error_code() -> int
    get_current_altitude() -> int
    get_current_airspeed() -> int
    get_is_landed() -> float
    get_euler_angles_of_camera() -> dict
    get_euler_angles_of_plane() -> dict
    get_gps_coordinates() -> dict

    set_gps_coordinates(gpsCoordinates: dict)
    set_ground_commands(groundCommands: dict)
    set_gimbal_commands(gimbalCommands: dict)
    set_begin_landing(beginLanding: bool)
    set_begin_takeoff(beginTakeoff: bool)
    set_disconnect_autopilot(disconnectAutoPilot: bool)

    @property pogiDirectory()
    @pogiDirectory.setter pogiDirectory(pogiFileDirectory: str)
    @property pigoDirectory()
    @pigoDirectory.setter pigoDirectory(pigoFileDirectory: str)
    """

    def __init__(self, pogiFileDirectory: str, pigoFileDirectory: str):
        """
        Initializes POGI & PIGO empty dictionaries, file directories, and file locks, and logger

        Parameters
        ----------
        pogiFileDirectory: str
            String of POGI file directory
        pigoFileDirectory: str
            String of POGI file directory
        """
        self.__logger = logging.getLogger(__name__)
        self.__pogiData = dict()
        self.__pigoData = dict()
        self.pogiFileDirectory = pogiFileDirectory
        self.pigoFileDirectory = pigoFileDirectory
        self.__pogiLock = FileLock(pogiFileDirectory + ".lock")
        self.__pigoLock = FileLock(pigoFileDirectory + ".lock")

    def __read_from_pogi_file(self):
        """
        Decodes JSON data from pogiFileDirectory JSON file and stores it in pogiData dictionary
        """
        try:
            with self.__pogiLock, open(self.pogiFileDirectory, "r") as pogiFile:
                self.__pogiData = json.load(pogiFile)
        except json.decoder.JSONDecodeError:
            self.__logger.error("The given POGI file has no data to read")
            return None

    def __write_to_pigo_file(self):
        """
        Encodes pigoData dictionary as JSON and stores it in pigoFileDirectory JSON file
        """
        if not bool(self.__pigoData):
            self.__logger.error("The current PIGO data dictionary is empty")
            return None
        with self.__pigoLock, open(self.pigoFileDirectory, "w") as pigoFile:
            json.dump(self.__pigoData, pigoFile, ensure_ascii=False, indent=4, sort_keys=True)

    def get_error_code(self) -> int:
        """
        Returns the error code from the POGI file

        Returns
        -------
        int:
            Error code
        """
        self.__read_from_pogi_file()
        errorCode = self.__pogiData["errorCode"]

        if errorCode is None:
            self.__logger.error("errorCode not found in the POGI json file.")
            return None
        if type(errorCode) is not int:
            self.__logger.error("errorCode found in POGI file is not an int.")
            return None
        
        return errorCode

    def get_current_altitude(self) -> int:
        """
        Returns the current altitude from the POGI file

        Returns
        -------
        int:
            Current altitude of the plane
        """
        self.__read_from_pogi_file()
        currentAltitude = self.__pogiData["currentAltitude"]

        if currentAltitude is None:
            self.__logger.error("currentAltitude not found in the POGI json file.")
            return None
        if type(currentAltitude) is not int:
            self.__logger.error("currentAltitude in the POGI file is not an int.")
            return None

        return currentAltitude

    def get_current_airspeed(self) -> int:
        """
        Returns the current airspeed from the POGI file

        Returns
        -------
        int:
            Current airspeed of the plane
        """
        self.__read_from_pogi_file()
        currentAirspeed = self.__pogiData["currentAirspeed"]

        if currentAirspeed is None:
            self.__logger.error("currentAirspeed not found in the POGI json file.")
            return None
        if type(currentAirspeed) is not int:
            self.__logger.error("currentAirspeed in the POGI file is not an int.")
            return None

        return currentAirspeed

    def get_is_landed(self) -> bool:
        """
        Returns if the plane has landed from the POGI file

        Returns
        -------
        bool:
            True if plane landed, else False
        """
        self.__read_from_pogi_file()
        isLanded = self.__pogiData["isLanded"]

        if isLanded is None:
            self.__logger.error("isLanded not found in the POGI json file.")
            return None
        if type(isLanded) is not bool:
            self.__logger.error("isLanded in the POGI file is not an int.")
            return None

        return isLanded

    def get_euler_angles_of_camera(self) -> dict:
        """
        Returns the euler coordinates of the camera on the plane from the POGI file

        Returns
        -------
        dict:
            Returns a dictionary that contains a set of three euler coordinates with names yaw, pitch, roll
        """
        self.__read_from_pogi_file()
        eulerAnglesOfCamera = self.__pogiData["eulerAnglesOfCamera"]

        if eulerAnglesOfCamera is None:
            self.__logger.error("eulerAnglesOfCamera not found in the POGI json file.")
            return None
        if type(eulerAnglesOfCamera) is not dict:
            self.__logger.error("eulerAnglesOfCamera in the POGI file is not a dictionary.")
            return None
        for key in ("yaw", "pitch", "roll"):
            if key not in eulerAnglesOfCamera.keys():
                self.__logger.error("{} key not found in eulerAnglesOfCamera.".format(key))
                return None
        for key in ("yaw", "pitch", "roll"):
            if eulerAnglesOfCamera[key] is None:
                self.__logger.error("{} in eulerAnglesOfCamera is null.".format(key))
                return None
        for key in ("yaw", "pitch", "roll"):
            if type(eulerAnglesOfCamera[key]) is not float:
                self.__logger.error("{} in eulerAnglesOfCamera is not a float.".format(key))
                return None

        return eulerAnglesOfCamera

    def get_euler_angles_of_plane(self) -> dict:
        """
        Returns the euler coordinates of the plane itself from the POGI file

        Returns
        -------
        dict:
            Returns a dictionary that contains a set of three euler coordinates with names yaw, pitch, roll
        """
        self.__read_from_pogi_file()
        eulerAnglesOfPlane = self.__pogiData["eulerAnglesOfPlane"]

        if eulerAnglesOfPlane is None:
            self.__logger.error("eulerAnglesOfPlane not found in the POGI json file.")
            return None
        if type(eulerAnglesOfPlane) is not dict:
            self.__logger.error("eulerAnglesOfPlane in the POGI file is not a dictionary.")
            return None
        for key in ("yaw", "pitch", "roll"):
            if key not in eulerAnglesOfPlane.keys():
                self.__logger.error("{} key not found in eulerAnglesOfPlane.".format(key))
                return None
        for key in ("yaw", "pitch", "roll"):
            if eulerAnglesOfPlane[key] is None:
                self.__logger.error("{} in eulerAnglesOfPlane is null.".format(key))
                return None
        for key in ("yaw", "pitch", "roll"):
            if type(eulerAnglesOfPlane[key]) is not float:
                self.__logger.error("{} in eulerAnglesOfPlane is not a float.".format(key))
                return None

        return eulerAnglesOfPlane

    def get_gps_coordinates(self) -> dict:
        """
        Returns the gps coordinates of the plane from the POGI file

        Returns
        -------
        dict:
            Returns a dictionary that contains GPS coordinate data with latitude, longitude, and altitude
        """
        self.__read_from_pogi_file()
        gpsCoordinates = self.__pogiData["gpsCoordinates"]

        if gpsCoordinates is None:
            self.__logger.error("gpsCoordinates not found in the POGI json file.")
            return None
        if type(gpsCoordinates) is not dict:
            self.__logger.error("gpsCoordinates in the POGI file is not a dictionary.")
            return None
        for key in ("latitude", "longitude", "altitude"):
            if key not in gpsCoordinates.keys():
                self.__logger.error("{} key not found in gpsCoordinates.".format(key))
                return None
        for key in ("latitude", "longitude", "altitude"):
            if gpsCoordinates[key] is None:
                self.__logger.error("{} in gpsCoordinates is null.".format(key))
                return None
        for key in ("latitude", "longitude", "altitude"):
            if type(gpsCoordinates[key]) is not float:
                self.__logger.error("{} in gpsCoordinates is not a float.".format(key))
                return None

        return gpsCoordinates

    def get_editing_flight_path_error_code(self) -> int:
        self.__read_from_pogi_file()
        error_code = self.__pogiData["editingFlightPathErrorCode"]
        if(error_code is None):
            self.__logger.error("editingFlightPathErrorCode must be an int and not None.")
            return None
        if type(error_code) is not int:
            self.__logger.error("editingFlightPathErrorCode is not an int.")
            return None

        return error_code

    def get_flight_path_following_error_code(self) -> int:
        self.__read_from_pogi_file()
        error_code = self.__pogiData["flightPathFollowingErrorCode"]
        if(error_code is None):
            self.__logger.error("flightPathFollowingErrorCode must be an int and not None.")
            return None
        if type(error_code) is not int:
            self.__logger.error("flightPathFollowingErrorCode is not an int.")
            return None

        return error_code

    def get_current_waypoint_id(self) -> int:
        self.__read_from_pogi_file()
        waypoint_id = self.__pogiData["currentWaypointId"]
        if(waypoint_id is None):
            self.__logger.error("currentWaypointId must be an int and not None.")
            return None
        if type(waypoint_id) is not int:
            self.__logger.error("currentWaypointId is not an int.")
            return None

        return waypoint_id

    def get_current_waypoint_index(self) -> int:
        self.__read_from_pogi_file()
        waypoint_index = self.__pogiData["currentWaypointIndex"]
        if(waypoint_index is None):
            self.__logger.error("currentWaypointIndex must be an int and not None.")
            return None
        if type(waypoint_index) is not int:
            self.__logger.error("currentWaypointIndex is not an int.")
            return None

        return waypoint_index

    def get_home_base_intialized(self) -> bool:
        self.__read_from_pogi_file()
        initialized = self.__pogiData["homeBaseInitialized"]
        if(initialized is None):
            self.__logger.error("homeBaseInitialized must be a bool and not None.")
            return None
        if type(initialized) is not bool:
            self.__logger.error("homeBaseInitialized is not a bool.")
            return None

        return initialized

    def set_gps_coordinates(self, gpsCoordinates: dict):
        """
        Write GPS coordinates to PIGO JSON file

        Paramaters
        ----------
        gpsCoordinates: dict
            Dictionary that contains GPS coordinate data with latitude, longitude, and altitude
        """
        if gpsCoordinates is None:
            self.__logger.error("gpsCoordinates must be a dict and not None.")
            return None
        if type(gpsCoordinates) is not dict:
            self.__logger.error("gpsCoordinates must be a dict and not {}.".format(type(gpsCoordinates)))
            return None
        for attribute in ("latitude", "longitude", "altitude"):
            if attribute not in gpsCoordinates.keys():
                self.__logger.error("gpsCoordinates must contain {} key.".format(attribute))
                return None
        for key in gpsCoordinates.keys():
            if type(gpsCoordinates[key]) is not float:
                self.__logger.error("gpsCoordinates {} key must be a float.".format(key))
                return None

        self.__pigoData.update({"gpsCoordinates" : gpsCoordinates})
        self.__write_to_pigo_file()

    def set_ground_commands(self, groundCommands: dict):
        """
        Write ground commands to PIGO JSON file

        Paramaters
        ----------
        groundCommands: dict
            Dictionary that contains ground commands with heading and latestDistance
        """
        if groundCommands is None:
            self.__logger.error("groundCommands must be a dict and not None.")
            return None
        if type(groundCommands) is not dict:
            self.__logger.error("groundCommands must be a dict and not {}.".format(type(groundCommands)))
            return None
        for key in ("heading", "latestDistance"):
            if key not in groundCommands.keys():
                self.__logger.error("groundCommands must contain {} key.".format(key))
                return None
        for key in ("heading", "latestDistance"):
            if type(groundCommands[key]) is not float:
                self.__logger.error("groundCommands {} key must be a float.".format(key))
                return None

        self.__pigoData.update({"groundCommands" : groundCommands})
        self.__write_to_pigo_file()

    def set_gimbal_commands(self, gimbalCommands: dict):
        """
        Write gimbal commands to PIGO JSON file

        Paramaters
        ----------
        gimbalCommands: dict
            Dictionary that contains gimbal commands with pitch and yaw angles
        """
        if gimbalCommands is None:
            self.__logger.error("gimbalCommands must be a dict and not None.")
            return None
        if type(gimbalCommands) is not dict:
            self.__logger.error("gimbalCommands must be a dict and not {}.".format(type(gimbalCommands)))
            return None
        for key in ("yaw", "pitch"):
            if key not in gimbalCommands.keys():
                self.__logger.error("gimbalCommands must contain {} key.".format(key))
                return None
        for key in ("yaw", "pitch"):
            if type(gimbalCommands[key]) is not float:
                self.__logger.error("gimbalCommands {} key must be a float.".format(key))
                return None

        self.__pigoData.update({"gimbalCommands" : gimbalCommands})
        self.__write_to_pigo_file()

    def set_begin_landing(self, beginLanding: bool):
        """
        Write begin landing command to PIGO JSON file

        Paramaters
        ----------
        beginLanding: bool
            Boolean containing whether or not to initiate landing sequence
        """
        if beginLanding is None:
            self.__logger.error("beginLanding must be a bool and not None.")
            return None
        if type(beginLanding) is not bool:
            self.__logger.error("beginLanding must be a bool and not {}.".format(type(beginLanding)))
            return None

        self.__pigoData.update({"beginLanding" : beginLanding})
        self.__write_to_pigo_file()

    def set_begin_takeoff(self, beginTakeoff: bool):
        """
        Write begin takeoff command to PIGO JSON file

        Paramaters
        ----------
        beginTakeoff: bool
            Boolean containing whether or not to initiate takeoff sequence
        """
        if beginTakeoff is None:
            self.__logger.error("beginTakeoff must be a bool and not None.")
            return None
        if type(beginTakeoff) is not bool:
            self.__logger.error("beginTakeoff must be a bool and not {}.".format(type(beginTakeoff)))
            return None

        self.__pigoData.update({"beginTakeoff" : beginTakeoff})
        self.__write_to_pigo_file()

    def set_num_waypoints(self, numWaypoints: int):
        """

        Parameters
        ----------
        numWaypoints: int

        """
        if numWaypoints is None:
            self.__logger.error("numWaypoints must be an int and not None.")
            return None
        if type(numWaypoints) is not int:
            self.__logger.error("numWaypoints must be an int and not {}.".format(type(numWaypoints)))
            return None
        self.__pigoData.update({"numWaypoints": numWaypoints})
        self.__write_to_pigo_file()

    def set_waypoint_modify_flight_path_command(self, waypointModifyFlightPathCommand: int):
        """

        Parameters
        ----------
        waypointModifyFlightPathCommand: int

        """
        if waypointModifyFlightPathCommand is None:
            self.__logger.error("waypointModifyFlightPathCommand must be an int and not None.")
            return None
        if type(waypointModifyFlightPathCommand) is not int:
            self.__logger.error("waypointModifyFlightPathCommand must be an int and not {}.".format(type(waypointModifyFlightPathCommand)))
            return None
        self.__pigoData.update({"waypointModifyFlightPathCommand": waypointModifyFlightPathCommand})
        self.__write_to_pigo_file()

    def set_waypoint_next_directions_command(self, waypointNextDirectionsCommand: int):
        """

        Parameters
        ----------
        waypointNextDirectionsCommand: int

        """
        if waypointNextDirectionsCommand is None:
            self.__logger.error("waypointNextDirectionsCommand must be an int and not None.")
            return None
        if type(waypointNextDirectionsCommand) is not int:
            self.__logger.error("waypointNextDirectionsCommand must be an int and not {}.".format(type(waypointNextDirectionsCommand)))
            return None
        self.__pigoData.update({"waypointNextDirectionsCommand": waypointNextDirectionsCommand})
        self.__write_to_pigo_file()

    def set_initializing_home_base(self, initializingHomeBase: bool):
        """

        Parameters
        ----------
        initializingHomeBase: bool

        """
        if initializingHomeBase is None:
            self.__logger.error("initializingHomeBase must be a bool and not None.")
            return None
        if type(initializingHomeBase) is not bool:
            self.__logger.error("initializingHomeBase must be a bool and not {}.".format(type(initializingHomeBase)))
            return None
        self.__pigoData.update({"initializingHomeBase": initializingHomeBase})
        self.__write_to_pigo_file()

    def set_flight_path_modify_next_id(self, flightPathModifyNextId: int):
        """

        Parameters
        ----------
        flightPathModifyNextId: int

        """
        if flightPathModifyNextId is None:
            self.__logger.error("flightPathModifyNextId must be an int and not None.")
            return None
        if type(flightPathModifyNextId) is not int:
            self.__logger.error("flightPathModifyNextId must be an int and not {}.".format(
                type(flightPathModifyNextId)))
            return None
        self.__pigoData.update({"flightPathModifyNextId": flightPathModifyNextId})
        self.__write_to_pigo_file()

    def set_flight_path_modify_prev_id(self, flightPathModifyPrevId: int):
        """

        Parameters
        ----------
        flightPathModifyPrevId: int

        """
        if flightPathModifyPrevId is None:
            self.__logger.error("flightPathModifyPrevId must be an int and not None.")
            return None
        if type(flightPathModifyPrevId) is not int:
            self.__logger.error("flightPathModifyPrevId must be an int and not {}.".format(
                type(flightPathModifyPrevId)))
            return None
        self.__pigoData.update({"flightPathModifyPrevId": flightPathModifyPrevId})
        self.__write_to_pigo_file()

    def set_flight_path_modify_id(self, flightPathModifyId: int):
        """

        Parameters
        ----------
        flightPathModifyId: int

        """
        if flightPathModifyId is None:
            self.__logger.error("flightPathModifyId must be an int and not None.")
            return None
        if type(flightPathModifyId) is not int:
            self.__logger.error("flightPathModifyId must be an int and not {}.".format(
                type(flightPathModifyId)))
            return None
        self.__pigoData.update({"flightPathModifyId": flightPathModifyId})
        self.__write_to_pigo_file()
    
    def set_holding_altitude(self, holdingAltitude: int):
        """

        Parameters
        ----------
        holdingAltitude: int

        """
        if holdingAltitude is None:
            self.__logger.error("holdingAltitude must be an int and not None.")
            return None
        if type(holdingAltitude) is not int:
            self.__logger.error("holdingAltitude must be an int and not {}.".format(
                type(holdingAltitude)))
            return None
        self.__pigoData.update({"holdingAltitude": holdingAltitude})
        self.__write_to_pigo_file()
    
    def set_holding_turn_radius(self, holdingTurnRadius: int):
        """

        Parameters
        ----------
        holdingTurnRadius: int

        """
        if holdingTurnRadius is None:
            self.__logger.error("holdingTurnRadius must be an int and not None.")
            return None
        if type(holdingTurnRadius) is not int:
            self.__logger.error("holdingTurnRadius must be an int and not {}.".format(
                type(holdingTurnRadius)))
            return None
        self.__pigoData.update({"holdingTurnRadius": holdingTurnRadius})
        self.__write_to_pigo_file()
    
    def set_holding_turn_direction(self, holdingTurnDirection: int):
        """

        Parameters
        ----------
        holdingTurnDirection: int

        """
        if holdingTurnDirection is None:
            self.__logger.error("holdingTurnDirection must be an int and not None.")
            return None
        if type(holdingTurnDirection) is not int:
            self.__logger.error("holdingTurnDirection must be an int and not {}.".format(
                type(holdingTurnDirection)))
            return None
        self.__pigoData.update({"holdingTurnDirection": holdingTurnDirection})
        self.__write_to_pigo_file()

    def set_waypoints(self, waypoints: np.ndarray):
        """
        Parameters
        ----------
        flightPathModifyId: np.array
            A numpy array that contains a dictionary housing latitude, longitude, altitude, turnRadius and waypointType

        """
        if waypoints is None:
            self.__logger.error("waypoints must be an array and not None.")
            return None
        if type(waypoints) is not np.ndarray:
            self.__logger.error("waypoints must be an array and not {}.".format(
                type(waypoints)))
            return None
        for waypoint in range(len(waypoints)):
            for key in ("latitude", "longitude", "altitude", "turnRadius", "waypointType"):
                if (waypoint[key] is None):
                    self.__logger.error("{} is None".format(key))
                    return None
            for key in ("latitude", "longitude"):
                if (type(waypoint[key]) is not float):
                    self.__logger.error("{} is {} and not a float".format(key, type(waypoint[key])))
                    return None
            for key in ("altitude", "turnRadius", "waypointType"):
                if(type(waypoint[key]) is not int):
                    self.__logger.error("{} is {} and not an int".format(key, type(waypoint[key])))
                    return None
                   
        self.__pigoData.update({"waypoints": waypoints})
        self.__write_to_pigo_file()

    def set_homebase(self, homebase: dict):
        """
        Write homebase to PIGO JSON file

        Paramaters
        ----------
        homebase: dict
            Dictionary that contains latitude (float), longitude (float), altitude (int), turnRadius (float), and waypointType (int) for homebase
        """

        if homebase is None:
            self.__logger.error("homebase must be a dict and not None.")
            return None
        if type(homebase) is not dict:
            self.__logger.error("homebase must be a dict and not {}.".format(type(homebase)))
            return None

        for key in ("latitude", "longitude", "altitude", "turnRadius", "waypointType"):
            if key not in homebase.keys():
                self.__logger.error("homebase must contain {} key.".format(key))
                return None
        for key in ("latitude", "longitude", "turnRadius"):
            if type(homebase[key]) is not float:
                self.__logger.error("homebase {} key must be a float.".format(key))
                return None
        for key in ("altitude", "waypointType"):
            if type(homebase[key]) is not int:
                self.__logger.error("homebase {} key must be an int.".format(key))
                return None

        self.__pigoData.update({"homebase": homebase})
        self.__write_to_pigo_file()

    def set_disconnect_autopilot(self, disconnectAutoPilot: bool):
        """
        Write disconnect autopilot to PIGO JSON file

        Paramaters
        ----------
        disconnectAutoPilot: bool
            Boolean containing whether or not to disconnect auto pilot
        """
        if disconnectAutoPilot is None:
            self.__logger.error("disconnectAutopilot must be a bool and not None.")
            return None
        if type(disconnectAutoPilot) is not bool:
            self.__logger.error("disconnectAutopilot must be a bool and not {}.".format(type(disconnectAutoPilot)))
            return None

        self.__pigoData.update({"disconnectAutoPilot" : disconnectAutoPilot})
        self.__write_to_pigo_file()

    @property
    def pigoFileDirectory(self):
        """
        Return PIGO JSON file directory

        Returns
        ----------
        __pigoFileDirectory: str
            Contains directory to PIGO JSON file
        """
        return self.__pigoFileDirectory

    @pigoFileDirectory.setter
    def pigoFileDirectory(self, pigoFileDirectory: str):
        """
        Sets PIGO JSON file directory

        Parameters
        ----------
        pigoFileDirectory: str
            Contains directory to PIGO JSON file
        """
        if pigoFileDirectory is None:
            raise ValueError("PIGO File Directory must be a str and not None.")
        if type(pigoFileDirectory) is not str:
            raise TypeError("PIGO File Directory must be a str and not {}".format(type(pigoFileDirectory)))
        if not os.path.isfile(pigoFileDirectory):
            raise FileNotFoundError("The PIGO file directory must be a valid file")
        if not pigoFileDirectory.endswith(".json"):
            raise ValueError("The PIGO file directory must have a '.json' file extension")
        self.__pigoFileDirectory = pigoFileDirectory

    @property
    def pogiFileDirectory(self):
        """
        Return POGI JSON file directory

        Returns
        ----------
        __pogiFileDirectory: str
            Contains directory to POGI JSON file
        """
        return self.__pogiFileDirectory

    @pogiFileDirectory.setter
    def pogiFileDirectory(self, pogiFileDirectory: str):
        """
        Sets POGI JSON file directory

        Parameters
        ----------
        pogiFileDirectory: str
            Contains directory to POGI JSON file
        """
        if pogiFileDirectory is None:
            raise ValueError("POGI File Directory must be a str and not None.")
        if type(pogiFileDirectory) is not str:
            raise TypeError("POGI File Directory must be a str and not {}".format(type(pogiFileDirectory)))
        if not os.path.isfile(pogiFileDirectory):
            raise FileNotFoundError("The POGI file directory must be a valid file")
        if not pogiFileDirectory.endswith(".json"):
            raise ValueError("The POGI file directory must have a '.json' file extension")
        self.__pogiFileDirectory = pogiFileDirectory