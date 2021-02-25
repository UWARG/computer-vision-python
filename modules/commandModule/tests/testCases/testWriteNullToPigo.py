from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingNullToPIGOFile(unittest.TestCase):
    """
    Test Case: Null written to PIGO file results in TypeError
    Methods to test:
    - set_gps_coordintes
	- set_ground_commands
	- set_gimbal_commands
	- set_begin_landing
	- set_begin_takeoff
	- set_disconnect_autopilot
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_type_error_if_set_gps_coords_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_gps_coordinates(None)
    
    def test_type_error_if_set_ground_commands_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_ground_commands(None)
    
    def test_type_error_if_set_gimbal_commands_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_gimbal_commands(None)
    
    def test_type_error_if_set_begin_landing_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_begin_landing(None)

    def test_type_error_if_set_begin_takeoff_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_begin_takeoff(None)

    def test_type_error_if_set_disconnect_autopilot_to_null(self):
        with self.assertRaises(TypeError):
            self.commandModule.set_disconnect_autopilot(None)