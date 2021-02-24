from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingWrongTypeToPIGOFile(unittest.TestCase):
    """
    Test Case: Wrong data type written to PIGO file results in sys.exit(1)
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
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "test.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile)

    def tearDown(self):
        self.testData = []
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def test_system_exit_if_set_gps_coords_to_wrong_type(self):
        for test in self.testData:
            with self.subTest(passed_data=test):
                with self.assertRaises(SystemExit) as cm:
                    self.commandModule.set_gps_coordinates(test)
                self.assertEqual(cm.exception.code, 1)
    
    def test_system_exit_if_set_ground_commands_to_wrong_type(self):
        for test in self.testData:
            with self.subTest(passed_data=test):
                with self.assertRaises(SystemExit) as cm:
                    self.commandModule.set_ground_commands(test)
                self.assertEqual(cm.exception.code, 1)
    
    def test_system_exit_if_set_gimbal_commands_to_wrong_type(self):
        for test in self.testData:
            with self.subTest(passed_data=test):
                with self.assertRaises(SystemExit) as cm:
                    self.commandModule.set_gimbal_commands(test)
                self.assertEqual(cm.exception.code, 1)
    
    def test_system_exit_if_set_begin_landing_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test):
                    with self.assertRaises(SystemExit) as cm:
                        self.commandModule.set_begin_landing(test)
                    self.assertEqual(cm.exception.code, 1)

    def test_system_exit_if_set_begin_takeoff_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test):
                    with self.assertRaises(SystemExit) as cm:
                        self.commandModule.set_begin_takeoff(test)
                    self.assertEqual(cm.exception.code, 1)

    def test_system_exit_if_set_disconnect_autopilot_to_wrong_type(self):
        for test in self.testData:
            if type(test) is not bool:
                with self.subTest(passed_data=test):
                    with self.assertRaises(SystemExit) as cm:
                        self.commandModule.set_disconnect_autopilot(test)
                    self.assertEqual(cm.exception.code, 1)