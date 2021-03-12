import json
import logging
import unittest
import os
from ...commandModule import CommandModule

class TestReadingWrongTypeFromPOGIFiles(unittest.TestCase):

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

    def test_type_error_if_get_error_code_equals_null(self):
        self.__value_instantiate("errorCode", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_error_code()

    def test_type_error_if_get_altitude_equals_wrong_type(self):
        self.__value_instantiate("currentAltitude", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_current_altitude()

    def test_type_error_if_get_airspeed_equals_wrong_type(self):
        self.__value_instantiate("currentAirspeed", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_current_airspeed()

    def test_type_error_if_get_is_landed_equals_wrong_type(self):
        self.__value_instantiate("isLanded", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_is_landed()

    def test_type_error_if_get_euler_angles_of_camera_equals_wrong_type(self):
        self.__value_instantiate("eulerAnglesOfCamera", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_camera()

    def test_type_error_if_get_euler_angles_of_camera_x_equals_wrong_type(self):
        eulerAnglesOfCamera = {"x": "0", "y": 0.0, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_camera()

    def test_type_error_if_get_euler_angles_of_camera_y_equals_wrong_type(self):
        eulerAnglesOfCamera = {"x": 0.0, "y": "0", "z": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_camera()

    def test_type_error_if_get_euler_angles_of_camera_z_equals_wrong_type(self):
        eulerAnglesOfCamera = {"x": 0.0, "y": 0.0, "z": "0"}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_camera()

    def test_type_error_if_get_euler_angles_of_plane_equals_wrong_type(self):
        self.__value_instantiate("eulerAnglesOfPlane", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_plane()

    def test_type_error_if_get_euler_angles_of_plane_x_equals_wrong_type(self):
        eulerAnglesOfCamera = {"x": "0", "y": 0.0, "z": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfCamera)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_plane()

    def test_type_error_if_get_euler_angles_of_plane_y_equals_wrong_type(self):
        eulerAnglesOfCamera = {"x": 0.0, "y": "0", "z": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfCamera)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_plane()

    def test_type_error_if_get_euler_angles_of_plane_z_equals_wrong_type(self):
        eulerAnglesOfPlane = {"x": 0.0, "y": 0.0, "z": "0"}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfPlane)
        with self.assertRaises(TypeError):
            self.commandModule.get_euler_angles_of_plane()

    def test_type_error_if_get_gps_equals_wrong_type(self):
        self.__value_instantiate("gpsCoordinates", "0")
        with self.assertRaises(TypeError):
            self.commandModule.get_gps_coordinates()

    def test_type_error_if_get_gps_latitude_equals_wrong_type(self):
        gps = {"latitude": "blah", "longitude": 12.31, "altitude": 11.23}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(TypeError):
            self.commandModule.get_gps_coordinates()

    def test_type_error_if_get_gps_longitude_equals_wrong_type(self):
        gps = {"latitude": 1.212, "longitude": "blah", "altitude": 11.23}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(TypeError):
            self.commandModule.get_gps_coordinates()
    
    def test_value_error_if_get_gps_alt_equals_wrong_type(self):
        gps = {"latitude": 1.212, "longitude": 12.31, "altitude": "blah"}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertRaises(TypeError):
            self.commandModule.get_gps_coordinates()