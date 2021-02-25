from ...commandModule import CommandModule
import unittest
import os

class TestCaseWritingMissingFileDirAsPIGOFileDir(unittest.TestCase):
    """
    Test Case: Missing file directory written as PIGO file directory results in sys.exit(1)
    Methods to test:
    - initializer
    """
    def setUp(self):
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
        self.pogiFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
        self.missingFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "oogaBooga.json")

    def test_system_exit_if_initialize_pogi_file_dir_as_missing_file_dir(self):
        with self.assertRaises(SystemExit) as cm:
            testCommandModule = CommandModule(pigoFileDirectory=self.pigoFile, pogiFileDirectory=self.missingFile)
        self.assertEqual(cm.exception.code, 1)
    
    def test_system_exit_if_initialize_pigo_file_dir_as_missing_file_dir(self):
        with self.assertRaises(SystemExit) as cm:
            testCommandModule = CommandModule(pigoFileDirectory=self.missingFile, pogiFileDirectory=self.pogiFile)
        self.assertEqual(cm.exception.code, 1)