import json
import logging
import unittest
import os
from modules.commandModule.commandModule import CommandModule
from modules.commandModule.tests.testCases.generate_temp_json import generate_temp_json

class TestReadingWrongTypeFromPOGIFiles(unittest.TestCase):

    def setUp(self):
        self.logger = logging.basicConfig(level=logging.DEBUG, )
        self.pogiData = dict()
        self.pigoFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json"))
        self.pogiFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json"))
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        self.pogiData = dict()
        os.remove(self.pigoFile)
        os.remove(self.pogiFile)

    def __value_instantiate(self, key, value):
        with open(self.pogiFile, "w") as file:
            temp = {key: value}
            json.dump(temp, file, ensure_ascii=False, indent=4, sort_keys=True)


    def test_type_error_if_get_altitude_equals_wrong_type(self):
        self.__value_instantiate("currentAltitude", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_current_altitude()
        self.assertEqual(cm.output, ["ERROR:root:currentAltitude in the POGI file is not an int.", ])
        logging.info(cm.output)

    def test_type_error_if_get_airspeed_equals_wrong_type(self):
        self.__value_instantiate("currentAirspeed", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_current_airspeed()
        self.assertEqual(cm.output, ["ERROR:root:currentAirspeed in the POGI file is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_is_landed_equals_wrong_type(self):
        self.__value_instantiate("isLanded", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_is_landed()
        self.assertEqual(cm.output, ["ERROR:root:isLanded in the POGI file is not an int.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_camera_equals_wrong_type(self):
        self.__value_instantiate("eulerAnglesOfCamera", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:root:eulerAnglesOfCamera in the POGI file is not a dictionary.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_camera_x_equals_wrong_type(self):
        eulerAnglesOfCamera = {"roll": "0", "pitch": 0.0, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:root:roll in eulerAnglesOfCamera is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_camera_y_equals_wrong_type(self):
        eulerAnglesOfCamera = {"roll": 0.0, "pitch": "0", "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:root:pitch in eulerAnglesOfCamera is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_camera_z_equals_wrong_type(self):
        eulerAnglesOfCamera = {"roll": 0.0, "pitch": 0.0, "yaw": "0"}
        self.__value_instantiate("eulerAnglesOfCamera", eulerAnglesOfCamera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_camera()
        self.assertEqual(cm.output, ["ERROR:root:yaw in eulerAnglesOfCamera is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_plane_equals_wrong_type(self):
        self.__value_instantiate("eulerAnglesOfPlane", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:root:eulerAnglesOfPlane in the POGI file is not a dictionary.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_plane_x_equals_wrong_type(self):
        eulerAnglesOfCamera = {"roll": "0", "pitch": 0.0, "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfCamera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:root:roll in eulerAnglesOfPlane is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_plane_y_equals_wrong_type(self):
        eulerAnglesOfCamera = {"roll": 0.0, "pitch": "0", "yaw": 0.0}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfCamera)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:root:pitch in eulerAnglesOfPlane is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_euler_angles_of_plane_z_equals_wrong_type(self):
        eulerAnglesOfPlane = {"roll": 0.0, "pitch": 0.0, "yaw": "0"}
        self.__value_instantiate("eulerAnglesOfPlane", eulerAnglesOfPlane)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_euler_angles_of_plane()
        self.assertEqual(cm.output, ["ERROR:root:yaw in eulerAnglesOfPlane is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_gps_equals_wrong_type(self):
        self.__value_instantiate("gpsCoordinates", "0")
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:root:gpsCoordinates in the POGI file is not a dictionary.", ])
        logging.info(cm.output)

    def test_type_error_if_get_gps_latitude_equals_wrong_type(self):
        gps = {"latitude": "blah", "longitude": 12.31, "altitude": 11.23}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:root:latitude in gpsCoordinates is not a float.", ])
        logging.info(cm.output)

    def test_type_error_if_get_gps_longitude_equals_wrong_type(self):
        gps = {"latitude": 1.212, "longitude": "blah", "altitude": 11.23}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:root:longitude in gpsCoordinates is not a float.", ])
        logging.info(cm.output)
    
    def test_value_error_if_get_gps_alt_equals_wrong_type(self):
        gps = {"latitude": 1.212, "longitude": 12.31, "altitude": "blah"}
        self.__value_instantiate("gpsCoordinates", gps)
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.get_gps_coordinates()
        self.assertEqual(cm.output, ["ERROR:root:altitude in gpsCoordinates is not a float.", ])
        logging.info(cm.output)


