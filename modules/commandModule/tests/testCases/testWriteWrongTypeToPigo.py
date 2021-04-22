from ...commandModule import CommandModule
import unittest
import os
import logging

class TestCaseWritingWrongTypeToPIGOFile(unittest.TestCase):
    """
    Test Case: Wrong data type written to PIGO file results in TypeError
    Methods to test:
    - set_gps_coordintes
	- set_ground_commands
	- set_gimbal_commands
	- set_begin_landing
	- set_begin_takeoff
	- set_disconnect_autopilot
    """
    def setUp(self):
        # store all python data types in a list
        self.testData = [str("Test"),
                         int(1),
                         float(3.14),
                         complex(2j),
                         list(("test1", "test2", "test3")),
                         tuple(("test1", "test2", "test3")),
                         range(6),
                         dict(key1="test1", key2=2.34),
                         set(("test1", "test2", "test3")),
                         frozenset(("test1", "test2", "test3")),
                         bool(0),
                         bytes(5),
                         bytearray(5),
                         memoryview(bytes(5))]
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        self.testData = []
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_type_error_if_set_gps_coords_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not dict:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_gps_coordinates(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:gpsCoordinates must be a dict and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_ground_commands_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not dict:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_ground_commands(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:groundCommands must be a dict and not {}.".format(type(test)), ])
                logging.info(cm.output)

    def test_type_error_if_set_gimbal_commands_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not dict:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_gimbal_commands(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:gimbalCommands must be a dict and not {}.".format(type(test)), ])
                logging.info(cm.output)

    def test_type_error_if_set_begin_landing_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_begin_landing(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:beginLanding must be a bool and not {}.".format(type(test)), ])
                logging.info(cm.output)

    def test_type_error_if_set_begin_takeoff_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_begin_takeoff(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:beginTakeoff must be a bool and not {}.".format(type(test)), ])
                logging.info(cm.output)

    def test_type_error_if_set_disconnect_autopilot_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_disconnect_autopilot(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:disconnectAutopilot must be a bool and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_initializing_home_base_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_initializing_home_base(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:initializingHomeBase must be a bool and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_num_waypoints_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_num_waypoints(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:numWaypoints must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_waypoint_modify_flight_path_command_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_waypoint_modify_flight_path_command(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:waypointModifyFlightPathCommand must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_waypoint_next_directions_command_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_waypoint_next_directions_command(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:waypointNextDirectionsCommand must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_flight_path_modify_next_id_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_flight_path_modify_next_id(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyNextId must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_flight_path_modify_prev_id_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_flight_path_modify_prev_id(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyPrevId must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_flight_path_modify_id_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_flight_path_modify_id(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:flightPathModifyId must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_homebase_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not dict:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_homebase(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:homebase must be a dict and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_holding_altitude(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_holding_altitude(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:holdingAltitude must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)
    
    def test_type_error_if_set_holding_turn_direction(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_holding_turn_direction(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:holdingTurnDirection must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)

    def test_type_error_if_set_holding_turn_radius(self):
        for test in self.testData:
            if type(test) is not int:
                with self.subTest(passed_data=test), self.assertLogs(level="ERROR") as cm:
                    self.commandModule.set_holding_turn_radius(test)
                self.assertEqual(cm.output, ["ERROR:commandModule.commandModule:holdingTurnRadius must be an int and not {}.".format(type(test)), ])
                logging.info(cm.output)