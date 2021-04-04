from ...commandModule import CommandModule
import unittest
import os
import logging

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
        temp = {"latestDistance":1.234}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(temp)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:groundCommands must contain heading key.", ])
        logging.info(cm.output)

    def test_key_error_if_set_ground_commands_missing_latest_distance_attribute(self):
        temp = {"heading":1.234}
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(temp)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:groundCommands must contain latestDistance key.", ])
        logging.info(cm.output)