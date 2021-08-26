import unittest
import os

from modules.commandModule.commandModule import CommandModule
from .generate_temp_json import generate_temp_json

class TestCaseWritingNullAsJSONFileDir(unittest.TestCase):
    """
    Test Case: Null written as JSON file directory results in ValueError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json"))
        self.pogiFile = generate_temp_json(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json"))

    def tearDown(self):
        os.remove(self.pigoFile)
        os.remove(self.pogiFile)

    def test_value_error_if_initialize_pogi_file_dir_to_none(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=None)
    
    def test_value_error_if_initialize_pigo_file_dir_to_none(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=None, pogiFileDirectory=self.pogiFile)