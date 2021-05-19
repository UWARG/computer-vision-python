import json
import logging
import unittest
import os
from ...commandModule import CommandModule

class TestReadingNullFromPOGIFiles(unittest.TestCase):

    def setUp(self):
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

    def test_value_error_if_get_value_error_if_get_altitude_equals_none(self):
        self.__value_instantiate("currentAltitude", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_current_altitude()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:currentAltitude not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_airspeed_equals_none(self):
        self.__value_instantiate("currentAirspeed", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_current_airspeed()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:currentAirspeed not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_is_landed_equals_none(self):
        self.__value_instantiate("isLanded", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_is_landed()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:isLanded not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_camera_equals_none(self):
        self.__value_instantiate("eulerAnglesOfCamera", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:eulerAnglesOfCamera not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_camera_x_equals_none(self):
        euler_camera = {"roll": None, "pitch": 0.0, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:roll in eulerAnglesOfCamera is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_camera_y_equals_none(self):
        euler_camera = {"roll": 0.0, "pitch": None, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:pitch in eulerAnglesOfCamera is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_camera_z_equals_none(self):
        euler_camera = {"roll": 0.0, "pitch": 0.0, "yaw": None}
        self.__value_instantiate("eulerAnglesOfCamera", euler_camera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:yaw in eulerAnglesOfCamera is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_plane_equals_none(self):
        self.__value_instantiate("eulerAnglesOfPlane", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:eulerAnglesOfPlane not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_plane_x_equals_none(self):
        euler_camera = {"roll": None, "pitch": 0.0, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", euler_camera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:roll in eulerAnglesOfPlane is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_plane_y_equals_none(self):
        euler_camera = {"roll": 0.0, "pitch": None, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", euler_camera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:pitch in eulerAnglesOfPlane is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_euler_plane_z_equals_none(self):
        euler_plane= {"roll": 0.0, "pitch": 0.0, "yaw": None}
        self.__value_instantiate("eulerAnglesOfPlane", euler_plane)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:yaw in eulerAnglesOfPlane is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_gps_equals_none(self):
        self.__value_instantiate("gpsCoordinates", None)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:gpsCoordinates not found in the POGI json file.", ])
        logging.info(cm.output)

    def test_value_error_if_get_gps_lat_equals_none(self):
        gps = {"latitude": None, "longitude": 2.134, "altitude": 1.234}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:latitude in gpsCoordinates is null.", ])
        logging.info(cm.output)

    def test_value_error_if_get_gps_lng_equals_none(self):
        gps = {"latitude": 1.212, "longitude": None, "altitude": 1.234}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:longitude in gpsCoordinates is null.", ])
        logging.info(cm.output)
    
    def test_value_error_if_get_gps_alt_equals_none(self):
        gps = {"latitude": 1.212, "longitude": 12.31, "altitude": None}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:altitude in gpsCoordinates is null.", ])
        logging.info(cm.output)

