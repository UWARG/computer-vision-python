import json
import logging
import unittest
import os
from ...commandModule import CommandModule


class TestReadingCorrectFromPOGIFiles(unittest.TestCase):

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

    def test_pass_if_get_error_code_equals_correct(self):
        self.__value_instantiate("errorCode", 0)
        self.assertEqual(0, self.commandModule.get_error_code())

    def test_pass_if_get_altitude_equals_correct(self):
        self.__value_instantiate("currentAltitude", 0)
        self.assertEqual(0, self.commandModule.get_current_altitude())

    def test_pass_if_get_airspeed_equals_correct(self):
        self.__value_instantiate("currentAirspeed", 0)
        self.assertEqual(0, self.commandModule.get_current_airspeed())

    def test_pass_if_get_is_landed_equals_correct(self):
        self.__value_instantiate("isLanded", True)
        self.assertEqual(True, self.commandModule.get_is_landed())

    def test_pass_if_get_euler_camera_equals_correct(self):
        euler_camera = {"x": 1.023, "y": 123.1, "z": 9.12}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        self.assertEqual(euler_camera, self.commandModule.get_euler_angles_of_camera())

    def test_pass_if_get_euler_plane_equals_correct(self):
        euler_plane = {"x": 1.023, "y": 123.1, "z": 9.12}
        self.__value_instantiate("eulerAnglesOfPlane", euler_plane)
        self.assertEqual(euler_plane, self.commandModule.get_euler_angles_of_plane())

    def test_pass_if_get_gps_equals_correct(self):
        gps = {"latitude": 1.212, "longitude": 2.134, "altitude": 1.234}
        self.__value_instantiate("gpsCoordinates", gps)
        self.assertEquals(gps, self.commandModule.get_gps_coordinates())