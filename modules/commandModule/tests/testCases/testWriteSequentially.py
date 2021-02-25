from ...commandModule import CommandModule
import unittest
import json
import os

"""
NOTE
This is supposed to be an integration test, but with no proper integration testing framework, I'm just doing it here for now
"""

class TestCaseWritingSequentialValuesToPIGOFile(unittest.TestCase):
    """
    Test Case: Multiple sequential writes to pigo file results in all data being stored properly
    Methods to test in sequential order: (note: random order)
    - set_gps_coordintes
    - set_begin_takeoff
    - set_begin_landing
	- set_ground_commands
	- set_disconnect_autopilot
    - set_gimbal_commands
    """

    def setUp(self):
        self.testData= {"gpsCoordinates": {"latitude": 2.34, "longitude": 1.34, "altitude": 1.278},
                        "groundCommands": {"heading": 1.23, "latestDistance": 2.34},
                        "gimbalCommands": {"pitch": 1.34, "yaw": 34.2},
                        "beginLanding": True,
                        "beginTakeoff": False,
                        "disconnectAutoPilot": False}
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def __read_json(self):
        # used to check file output
        with open(self.pigoFile, "r") as pigoFile:
            results = json.load(pigoFile)
        return results

    def test_correct_data_storage_if_write_data_sequentially(self):
        self.commandModule.set_gps_coordinates(self.testData["gpsCoordinates"])
        self.commandModule.set_begin_takeoff(self.testData["beginTakeoff"])
        self.commandModule.set_begin_landing(self.testData["beginLanding"])
        self.commandModule.set_ground_commands(self.testData["groundCommands"])
        self.commandModule.set_disconnect_autopilot(self.testData["disconnectAutoPilot"])
        self.commandModule.set_gimbal_commands(self.testData["gimbalCommands"])
        self.assertEqual(self.__read_json(), self.testData)