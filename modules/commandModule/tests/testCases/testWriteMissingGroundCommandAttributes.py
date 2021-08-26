import unittest
import os
import logging

from modules.commandModule.commandModule import CommandModule
from .generate_temp_json import generate_temp_json

class TestCaseWritingMissingGroundCommandAttributes(unittest.TestCase):
    """
    Test Case: Ground commands dict to write have missing attributes
    Methods to test:
	- set_ground_commands
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

    def test_key_error_if_set_ground_commands_missing_heading_attribute(self):
        temp = {"latestDistance":1.234}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(temp)
        self.assertEqual(cm.output, ["ERROR:root:groundCommands must contain heading key.", ])
        logging.info(cm.output)

    def test_key_error_if_set_ground_commands_missing_latest_distance_attribute(self):
        temp = {"heading":1.234}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(temp)
        self.assertEqual(cm.output, ["ERROR:root:groundCommands must contain latestDistance key.", ])
        logging.info(cm.output)