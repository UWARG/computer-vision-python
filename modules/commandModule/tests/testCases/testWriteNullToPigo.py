from ...commandModule import CommandModule
import unittest
import os
import logging

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
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_value_error_if_set_gps_coords_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gps_coordinates(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:gpsCoordinates must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_ground_commands_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_ground_commands(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:groundCommands must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_gimbal_commands_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_gimbal_commands(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:gimbalCommands must be a dict and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_begin_landing_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_begin_landing(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:beginLanding must be a bool and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_begin_takeoff_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_begin_takeoff(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:beginTakeoff must be a bool and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_disconnect_autopilot_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_disconnect_autopilot(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:disconnectAutopilot must be a bool and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_initializing_home_base_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_initializing_home_base(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:initializingHomeBase must be a bool and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_num_waypoints_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_num_waypoints(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:numWaypoints must be an int and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_waypoint_modify_flight_path_command_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_waypoint_modify_flight_path_command(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:waypointModifyFlightPathCommand must be an int and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_waypoint_next_directions_command_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_waypoint_next_directions_command(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:waypointNextDirectionsCommand must be an int and not None.", ])
        logging.info(cm.output)

    def test_value_error_if_set_flight_path_modify_next_id_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_flight_path_modify_next_id(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyNextId must be an int and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_flight_path_modify_prev_id_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_flight_path_modify_prev_id(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyPrevId must be an int and not None.", ])
        logging.info(cm.output)
    
    def test_value_error_if_set_flight_path_modify_id_to_null(self):
        with self.assertLogs(level="ERROR") as cm:
            self.commandModule.set_flight_path_modify_id(None)
        self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyId must be an int and not None.", ])
        logging.info(cm.output)