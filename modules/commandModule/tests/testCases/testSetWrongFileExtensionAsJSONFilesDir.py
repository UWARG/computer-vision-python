from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingNonJSONFileDirAsJSONFileDir(unittest.TestCase):
    """
    Test Case: Non-json file directory written as JSON file directory results in ValueError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.nonJSONFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "wrongType.txt")

    def test_value_error_if_initialize_pogi_file_dir_as_non_json_file(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.nonJSONFile)

    def test_value_error_if_initialize_pigo_file_dir_as_non_json_file(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=self.nonJSONFile, pogiFileDirectory=self.pogiFile)