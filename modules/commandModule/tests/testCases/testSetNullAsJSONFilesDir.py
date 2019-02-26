from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingNullAsPIGOFileDir(unittest.TestCase):
    """
    Test Case: Null written as PIGO file directory results in ValueError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")

    def test_value_error_if_initialize_pogi_file_dir_to_none(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=None)
    
    def test_value_error_if_initialize_pigo_file_dir_to_none(self):
        with self.assertRaises(ValueError):
            testCommandModule = CommandModule(pigoFileDirectory=None, pogiFileDirectory=self.pogiFile)