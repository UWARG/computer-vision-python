from ...commandModule import CommandModule
import unittest
import json
import os

class TestCaseWritingCorrectValuesToPIGOFile(unittest.TestCase):
    """
    Test Case: Correct values written to PIGO file results in pass
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

    def __read_json(self, key):
        # used to check file output
        with open(self.pigoFile, "r") as pigoFile:
            results = json.load(pigoFile)
        return results[key]

    def test_pass_if_write_correct_gps_coords(self):
        example = {"longitude": 2.34, "latitude": 1.28, "altitude": 1.72} # correct value is a dict containing long., lat., and alt. floats
        self.commandModule.set_gps_coordinates(example)
        self.assertEqual(self.__read_json("gpsCoordinates"), example)
    
    def test_pass_if_write_correct_ground_commands(self):
        example = {"heading": 2.34, "latestDistance": 1.28} # correct value is a dict containing heading and latestDistance floats
        self.commandModule.set_ground_commands(example)
        self.assertEqual(self.__read_json("groundCommands"), example)

    def test_pass_if_write_correct_gimbal_commands(self):
        example = {"pitch": 2.34, "yaw": 1.28}    # correct value is a dict containing pitch and yaw floats
        self.commandModule.set_gimbal_commands(example)
        self.assertEqual(self.__read_json("gimbalCommands"), example)
    
    def test_pass_if_write_correct_begin_landing(self):
        example = True  # correct value is a bool
        self.commandModule.set_begin_landing(example)
        self.assertEqual(self.__read_json("beginLanding"), example)
    
    def test_pass_if_write_correct_begin_takeoff(self):
        example = True  # correct value is a bool
        self.commandModule.set_begin_takeoff(example)
        self.assertEqual(self.__read_json("beginTakeoff"), example)
    
    def test_pass_if_write_correct_disconnect_autopilot(self):
        example = True  # correct value is a bool
        self.commandModule.set_disconnect_autopilot(example)
        self.assertEqual(self.__read_json("disconnectAutoPilot"), example)
    
    def test_pass_if_write_correct_disconnect_autopilot(self):
        example = True  # correct value is a bool
        self.commandModule.set_initializing_home_base(example)
        self.assertEqual(self.__read_json("initializingHomeBase"), example)
    
    def test_pass_if_write_correct_num_waypoints(self):
        example = 1  # correct value is an int
        self.commandModule.set_num_waypoints(example)
        self.assertEqual(self.__read_json("numWaypoints"), example)
    
    def test_pass_if_write_correct_waypoint_modify_flight_path_command(self):
        example = 1  # correct value is an int
        self.commandModule.set_waypoint_modify_flight_path_command(example)
        self.assertEqual(self.__read_json("waypointModifyFlightPathCommand"), example)
    
    def test_pass_if_write_correct_waypoint_next_directions_command(self):
        example = 1  # correct value is an int
        self.commandModule.set_waypoint_next_directions_command(example)
        self.assertEqual(self.__read_json("waypointNextDirectionsCommand"), example)
    
    def test_pass_if_write_correct_flight_path_modify_next_id(self):
        example = 1  # correct value is an int
        self.commandModule.set_flight_path_modify_next_id(example)
        self.assertEqual(self.__read_json("flightPathModifyNextId"), example)
    
    def test_pass_if_write_correct_flight_path_modify_prev_id(self):
        example = 1  # correct value is an int
        self.commandModule.set_flight_path_modify_prev_id(example)
        self.assertEqual(self.__read_json("flightPathModifyPrevId"), example)
    
    def test_pass_if_write_correct_flight_path_modify_id(self):
        example = 1  # correct value is an int
        self.commandModule.set_flight_path_modify_id(example)
        self.assertEqual(self.__read_json("flightPathModifyId"), example)
    
    def test_pass_if_write_correct_holding_turn_radius(self):
        example = 1  # correct value is an int
        self.commandModule.set_holding_turn_radius(example)
        self.assertEqual(self.__read_json("holdingTurnRadius"), example)
    
    def test_pass_if_write_correct_holding_turn_direction(self):
        example = 1  # correct value is an int
        self.commandModule.set_holding_turn_direction(example)
        self.assertEqual(self.__read_json("holdingTurnDirection"), example)
    
    def test_pass_if_write_correct_holding_altitude(self):
        example = 1  # correct value is an int
        self.commandModule.set_holding_altitude(example)
        self.assertEqual(self.__read_json("holdingAltitude"), example)
    
    def test_pass_if_write_correct_homebase(self):
        example = {"latitude": 1.23, "longitude": 1.23, "altitude": 1, "turnRadius": 1.23, "waypointType": 5}  # correct value is an int
        self.commandModule.set_homebase(example)
        self.assertEqual(self.__read_json("homebase"), example)