import unittest
import os

from modules.commandModule.commandModule import CommandModule
from .generate_temp_json import generate_temp_json

class TestCaseWritingNonFileDirAsJSONFileDir(unittest.TestCase):
    """
    Test Case: Non-file directory written as JSON file directory results in FileNotFoundError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json"))
        self.pogiFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json"))
        self.nonFileDir = os.path.dirname(os.path.realpath(__file__))

    def tearDown(self):
        os.remove(self.pigoFile)
        os.remove(self.pogiFile)

    def test_file_not_found_error_if_initialize_pogi_file_dir_as_non_file(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.nonFileDir)
    
    def test_file_not_found_error_if_initialize_pigo_file_dir_as_non_file(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.nonFileDir, pogiFileDirectory=self.pogiFile)