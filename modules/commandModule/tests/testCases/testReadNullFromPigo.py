from ...commandModule import CommandModule
import unittest
import logging
import json

class TestCaseReadingNullValuesFromPIGOFiles(unittest.TestCase):

    def setUp(self):
        self.logger = logging.basicConfig(level=logging.DEBUG,)
        self.pigoData = dict()
        self.pigoFile = str(__file__).replace("testWriteNullToPigo.py", "") + "../testJSONs/test.json"
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile)

    def tearDown(self):
        self.pigoData = dict()

    def test_system_exit_if_set_gps_coords_to_null(self):
        with self.assertRaises(SystemExit) as cm:
            self.commandModule.set_gps_coordinates(None)
        self.assertEqual(cm.exception.code, 1)
    
    def test_error_message_if_set_gps_coords_to_null(self):
        with self.assertLogs(logger=self.logger, level="ERROR") as cm:
            try:
                self.commandModule.set_gps_coordinates(None)
            except SystemExit:
                pass

        self.assertEqual(cm.output, ["ERROR:root:Value that was passed is null."])