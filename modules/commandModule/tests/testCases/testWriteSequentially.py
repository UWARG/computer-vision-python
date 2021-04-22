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
                        "gimbalCommands": {"yaw": 1.34, "pitch": 34.2},
                        "beginLanding": True,
                        "beginTakeoff": False,
                        "disconnectAutoPilot": False,
                        "numWaypoints": 1,
                        "waypointModifyFlightPathCommand": 1,
                        "waypointNextDirectionsCommand": 1,
                        "flightPathModifyNextId": 1,
                        "flightPathModifyPrevId": 1,
                        "flightPathModifyId": 1,
                        "initializingHomeBase": True,
                        "homebase": {"latitude": 1.23, "longitude": 1.23, "altitude": 1, "turnRadius": 1.23, "waypointType": 5},
                        "holdingTurnDirection": 1,
                        "holdingTurnRadius": 1,
                        "holdingAltitude": 1}
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
        self.commandModule.set_num_waypoints(self.testData["numWaypoints"])
        self.commandModule.set_waypoint_modify_flight_path_command(self.testData["waypointModifyFlightPathCommand"])
        self.commandModule.set_waypoint_next_directions_command(self.testData["waypointNextDirectionsCommand"])
        self.commandModule.set_flight_path_modify_next_id(self.testData["flightPathModifyNextId"])
        self.commandModule.set_flight_path_modify_prev_id(self.testData["flightPathModifyPrevId"])
        self.commandModule.set_flight_path_modify_id(self.testData["flightPathModifyId"])
        self.commandModule.set_initializing_home_base(self.testData["initializingHomeBase"])
        self.commandModule.set_homebase(self.testData["homebase"])
        self.commandModule.set_holding_turn_direction(self.testData["holdingTurnDirection"])
        self.commandModule.set_holding_turn_radius(self.testData["holdingTurnRadius"])
        self.commandModule.set_holding_altitude(self.testData["holdingAltitude"])

        self.assertEqual(self.__read_json(), self.testData)