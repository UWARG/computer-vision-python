from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingMissingGPSCoordinateAttributes(unittest.TestCase):
    """
    Test Case: GPS coordinates dict to write have missing attributes
    Methods to test:
	- set_gps_coordinates
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_key_error_if_set_gps_coordinates_missing_latitude_attribute(self):
        with self.assertRaises(KeyError):
            self.commandModule.set_gps_coordinates(dict(longitude=1.234, altitude=2.312))

    def test_key_error_if_set_gps_coordinates_missing_longitude_attribute(self):
        with self.assertRaises(KeyError):
            self.commandModule.set_gps_coordinates(dict(latitude=1.234, altitude=2.312))

    def test_key_error_if_set_gps_coordinates_missing_altitude_attribute(self):
        with self.assertRaises(KeyError):
            self.commandModule.set_gps_coordinates(dict(latitude=1.234, longitude=2.312))