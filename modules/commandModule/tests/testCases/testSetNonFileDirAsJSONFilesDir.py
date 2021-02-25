from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingNonFileDirAsPIGOFileDir(unittest.TestCase):
    """
    Test Case: Non-file directory written as PIGO file directory results in FileNotFoundError
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.nonFileDir = os.path.dirname(os.path.realpath(__file__))

    def test_file_not_found_error_if_initialize_pogi_file_dir_as_non_file(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.nonFileDir)
    
    def test_file_not_found_error_if_initialize_pigo_file_dir_as_non_file(self):
        with self.assertRaises(FileNotFoundError):
            testCommandModule = CommandModule(pigoFileDirectory=self.nonFileDir, pogiFileDirectory=self.pogiFile)