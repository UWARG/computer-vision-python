import random
import json
import math

class mockCommandModule:
    """
    Mock Commmand Module that generates random POGI file for testing

    Methods
    -------
    __init__(pogiFileDirectory)
    get_gps_coordinates() -> dict
    get_current_airspeed() -> dict
    get_euler_angles_of_plane() -> dict
    get_euler_angles_of_camera() -> dict
    get_is_landed() -> dict
    write_to_json() -> file
    create_json() -> file

    """

    def __init__(self, pogiFileDirectory: str):
        self.pogiFileDirectory = pogiFileDirectory

    def get_gps_coordinates():
        return {
            'latitude': random.uniform(-90, 90),
            'longitude': random.uniform(-180, 180),
            'altitude': random.uniform(0, 100)
        }

    def get_current_airspeed():
        return {
            round(random.uniform(0, 50), 7)
        }

    def get_euler_angles_of_plane():
        return {
            'yaw': random.uniform(-90, 90),
            'pitch': random.uniform(-90, 90),
            'roll': random.uniform(-90, 90)
        }
    
    def get_euler_angles_of_camera():
        return {
            'yaw': random.uniform(-90, 90),
            'pitch': random.uniform(-90, 90),
            'roll': random.uniform(-90, 90)
        }

    def get_is_landed():
        return {
            random.randint(0,1)
        }

    def write_to_json(self):
        with open(self.pogiFileDirectory, 'w') as file:
            json.dump(self.__pogiData, file, ensure_ascii=False, indent=4, sort_keys=True)

    def create_json(self):
        pogi = {
            'gpsCoordinates': self.get_gps_coordinates(),
            'currentAirspeed': self.get_current_airspeed(),
            'eulerPlane': self.get_euler_angles_of_plane(),
            'eulerCamera': self.get_euler_angles_of_camera(),
            'isLanded': self.get_is_landed()
        }
        self.write_to_json(pogi)
        return pogi
