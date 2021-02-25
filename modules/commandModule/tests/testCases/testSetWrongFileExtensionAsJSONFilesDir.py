from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingNonJSONFileDirAsPIGOFileDir(unittest.TestCase):
    """
    Test Case: Non-json file directory written as PIGO file directory results in sys.exit(1)
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.nonJSONFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "wrongType.txt")

    def test_system_exit_if_initialize_pogi_file_dir_as_non_json_file(self):
        with self.assertRaises(SystemExit) as cm:
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.nonJSONFile)
        self.assertEqual(cm.exception.code, 1)
    
    def test_system_exit_if_initialize_pigo_file_dir_as_non_json_file(self):
        with self.assertRaises(SystemExit) as cm:
            testCommandModule = CommandModule(pigoFileDirectory=self.nonJSONFile, pogiFileDirectory=self.pogiFile)
        self.assertEqual(cm.exception.code, 1)