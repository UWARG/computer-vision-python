from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingMissingGimbalCommandAttributes(unittest.TestCase):
    """
    Test Case: Gimbal commands dict to write have missing attributes
    Methods to test:
	- set_gimbal_commands
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_key_error_if_set_gimbal_commands_missing_yaw_attribute(self):
        temp = {"y":1.231}
        with self.assertRaises(KeyError):
            self.commandModule.set_gimbal_commands(temp)

    def test_key_error_if_set_gimbal_commands_missing_pitch_attribute(self):
        temp = {"z":1.213}
        with self.assertRaises(KeyError):
            self.commandModule.set_gimbal_commands(temp)