from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingMissingFileDirAsJSONFileDir(unittest.TestCase):
    """
    Test Case: Missing file directory written as JSON file directory results in FileNotFoundError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.missingFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "oogaBooga.json")

    def test_file_not_found_error_if_initialize_pogi_file_dir_as_missing_file_dir(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.missingFile)
    
    def test_file_not_found_error_if_initialize_pigo_file_dir_as_missing_file_dir(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.missingFile, pogiFileDirectory=self.pogiFile)