import unittest
import os
import logging

from modules.commandModule.commandModule import CommandModule
from .generate_temp_json import generate_temp_json

class TestCaseWritingMissingGimbalCommandAttributes(unittest.TestCase):
    """
    Test Case: Gimbal commands dict to write have missing attributes
    Methods to test:
	- set_gimbal_commands
    """
    def setUp(self):
        self.pigoFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json"))
        self.pogiFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json"))
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        os.remove(self.pigoFile)
        os.remove(self.pogiFile)

    def test_key_error_if_set_gimbal_commands_missing_yaw_attribute(self):
        temp = {"pitch": 1.231}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gimbal_commands(temp)
        self.assertEqual(cm.output, ["ERROR:root:gimbalCommands must contain yaw key.", ])
        logging.info(cm.output)

    def test_key_error_if_set_gimbal_commands_missing_pitch_attribute(self):
        temp = {"yaw": 1.213}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gimbal_commands(temp)
        self.assertEqual(cm.output, ["ERROR:root:gimbalCommands must contain pitch key.", ])
        logging.info(cm.output)