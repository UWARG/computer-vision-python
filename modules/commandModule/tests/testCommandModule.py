import random
import json
import math

class mockCommandModule:
    """
    Mock Commmand Module that generates random POGI file for testing

    ...

    Attributes
    ----------

    pogiFileDirectory : str
        string containing directory to POGI file
    __pogiData : dict
        dictionary of POGI data to be written to POGI file
    __pogiData['gpsCoordinates'] : list
        list containing, in order, random latitude, longitude, altitude
    __pogiData['currentAirspeed] : list
        list containing random airspeed in m/s
    __pogiData['eulerAnglesOfPlane] : list
        list containing, in order, random yaw, pitch, row angles of plane in radians
    __pogiData['eulerAnglesOfCamera'] : list
        list containing, in order, random yaw, pitch, row angles of camera in radians
    __pogiData['isLanded'] : int
        list containing either 1 for True or 0 for False to indicate whether drone has landed
    """

    def __init__(self, pogiFileDirectory: str):
        self.pogiFileDirectory = pogiFileDirectory
        self.__pogiData = dict()
        self.__pogiData['gpsCoordinates'] = {'latitude': random.uniform(-90, 90), 'longitude': random.uniform(-180, 180), 'altitude': random.uniform(0, 100)}
        self.__pogiData['currentAirspeed'] = [round(random.uniform(0, 50), 7)]
        self.__pogiData['eulerAnglesOfPlane'] = {'yaw': random.uniform(0, math.pi * 2), 'pitch': random.uniform(-math.pi/2, math.pi/2), 'roll': random.uniform(-math.pi/2, math.pi/2)}
        self.__pogiData['eulerAnglesOfCamera'] = {'yaw': random.uniform(0, math.pi * 2), 'pitch': random.uniform(-math.pi/2, math.pi/2), 'roll': random.uniform(-math.pi/2, math.pi/2)}
        self.__pogiData['isLanded'] = [random.randint(0,1)]

    def write_to_json(self):
        with open(self.pogiFileDirectory, 'w') as file:
            json.dump(self.__pogiData, file, ensure_ascii=False, indent=4, sort_keys=True)

mockData = mockCommandModule("./randomPogi.json")
mockData.write_to_json()
