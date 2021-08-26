import unittest
import os

from modules.commandModule.commandModule import CommandModule
from modules.commandModule.tests.testCases.generate_temp_json import generate_temp_json

class TestCaseWritingWrongTypeToJSONFileDirectories(unittest.TestCase):
    """
    Test Case: Wrong data type used to set PIGO and POGI File Directories results in TypeError
    Methods to test:
    - initializer
    """
    def setUp(self):
        # store wrong data types in a list
        self.testData = [int(1),
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
        self.pigoFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json"))
        self.pogiFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json"))

    def tearDown(self):
        self.testData = []
        os.remove(self.pigoFile)
        os.remove(self.pogiFile)

    def test_type_error_if_initialize_pigo_file_dir_to_wrong_type(self):
        for test in self.testData:
            with self.subTest(passed_data=test):
                with self.assertRaises(TypeError): 
                    testCommandModule = CommandModule(pigoFileDirectory=test, pogiFileDirectory=self.pogiFile)
    
    def test_type_error_if_initialize_pogi_file_dir_to_wrong_type(self):
        for test in self.testData:
            with self.subTest(passed_data=test):
                with self.assertRaises(TypeError):
                    testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=test)