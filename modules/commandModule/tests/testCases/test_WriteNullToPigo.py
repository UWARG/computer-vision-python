from modules.commandModule.commandModule import CommandModule
import unittest
import os
import logging
import os

from modules.commandModule.tests.testCases.generate_temp_json import generate_temp_json

class TestCaseWritingNullToPIGOFile(unittest.TestCase):
    """
    Test Case: Null written to PIGO file results in ValueError
    Methods to test:
    - set_gps_coordintes
	- set_ground_commands
	- set_gimbal_commands
	- set_begin_landing
	- set_begin_takeoff
	- set_disconnect_autopilot
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

    def test_value_error_if_set_gps_coords_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gps_coordinates(None)
        self.assertEqual(cm.output, ["ERROR:root:gpsCoordinates must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_ground_commands_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(None)
        self.assertEqual(cm.output, ["ERROR:root:groundCommands must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_gimbal_commands_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gimbal_commands(None)
        self.assertEqual(cm.output, ["ERROR:root:gimbalCommands must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_begin_landing_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_begin_landing(None)
        self.assertEqual(cm.output, ["ERROR:root:beginLanding must be a bool and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_begin_takeoff_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_begin_takeoff(None)
        self.assertEqual(cm.output, ["ERROR:root:beginTakeoff must be a bool and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_disconnect_autopilot_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_disconnect_autopilot(None)
        self.assertEqual(cm.output, ["ERROR:root:disconnectAutopilot must be a bool and not None.", ])
        logging.info(cm.output)
