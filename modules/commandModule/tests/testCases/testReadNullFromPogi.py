import json
import logging
import unittest
import os
from ...commandModule import CommandModule

class TestReadingNullFromPOGIFiles(unittest.TestCase):

    def setUp(self):
        self.logger = logging.basicConfig(level=logging.DEBUG, )
        self.pogiData = dict()
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        self.pogiData = dict()

    def __value_instantiate(self, key, value):
        with open(self.pogiFile, "w") as file:
            temp = {key: value}
            json.dump(temp, file, ensure_ascii=False, indent=4, sort_keys=True)

    def test_value_error_if_get_error_code_equals_null(self):
        self.__value_instantiate("errorCode", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_error_code()

    def test_value_error_if_get_value_error_if_get_altitude_equals_none(self):
        self.__value_instantiate("currentAltitude", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_current_altitude()

    def test_value_error_if_get_airspeed_equals_none(self):
        self.__value_instantiate("currentAirspeed", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_current_airspeed()

    def test_value_error_if_get_is_landed_equals_none(self):
        self.__value_instantiate("isLanded", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_is_landed()

    def test_value_error_if_get_euler_camera_equals_none(self):
        self.__value_instantiate("eulerAnglesOfCamera", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_camera()

    def test_value_error_if_get_euler_camera_x_equals_none(self):
        euler_camera = {"x": None, "y": 0.0, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_camera()

    def test_value_error_if_get_euler_camera_y_equals_none(self):
        euler_camera = {"x": 0.0, "y": None, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_camera()

    def test_value_error_if_get_euler_camera_z_equals_none(self):
        euler_camera = {"x": 0.0, "y": 0.0, "z": None}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_camera()

    def test_value_error_if_get_euler_plane_equals_none(self):
        self.__value_instantiate("eulerAnglesOfPlane", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_plane()

    def test_value_error_if_get_euler_plane_x_equals_none(self):
        euler_camera = {"x": None, "y": 0.0, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", euler_camera)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_plane()

    def test_value_error_if_get_euler_plane_y_equals_none(self):
        euler_camera = {"x": 0.0, "y": None, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", euler_camera)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_plane()

    def test_value_error_if_get_euler_plane_z_equals_none(self):
        euler_plane= {"x": 0.0, "y": 0.0, "z": None}
        self.__value_instantiate("eulerAnglesOfPlane", euler_plane)
        with self.assertRaises(ValueError):
            self.commandModule.get_euler_angles_of_plane()

    def test_value_error_if_get_gps_equals_none(self):
        self.__value_instantiate("gpsCoordinates", None)
        with self.assertRaises(ValueError):
            self.commandModule.get_gps_coordinates()

    def test_value_error_if_get_gps_lat_equals_none(self):
        gps = {"latitude": None, "longitude": 2.134, "altitude": 1.234}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(ValueError):
            self.commandModule.get_gps_coordinates()

    def test_value_error_if_get_gps_lng_equals_none(self):
        gps = {"latitude": 1.212, "longitude": None, "altitude": 1.234}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(ValueError):
            self.commandModule.get_gps_coordinates()
    
    def test_value_error_if_get_gps_alt_equals_none(self):
        gps = {"latitude": 1.212, "longitude": 12.31, "altitude": None}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(ValueError):
            self.commandModule.get_gps_coordinates()