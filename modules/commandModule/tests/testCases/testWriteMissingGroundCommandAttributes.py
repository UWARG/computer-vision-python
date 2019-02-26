from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingMissingGroundCommandAttributes(unittest.TestCase):
    """
    Test Case: Ground commands dict to write have missing attributes
    Methods to test:
	- set_ground_commands
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_key_error_if_set_ground_commands_missing_heading_attribute(self):
        with self.assertRaises(KeyError):
            self.commandModule.set_gimbal_commands(dict(latestDistance=1.234))

    def test_key_error_if_set_ground_commands_missing_latest_distance_attribute(self):
        with self.assertRaises(KeyError):
            self.commandModule.set_gimbal_commands(dict(heading=1.234))