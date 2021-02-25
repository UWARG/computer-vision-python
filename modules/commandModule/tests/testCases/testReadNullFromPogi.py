import logging
import unittest
from ...commandModule import CommandModule

class TestReadingNullFromPOGIFiles(unittest.TestCase):

    def setUp(self):
        self.logger = logging.basicConfig(level=logging.DEBUG, )
        self.pogiData = dict()
        self.pogiFile = str(__file__).replace("testWriteNullToPogi.py", "") + "../testJSONs/test.json"
        self.commandModule = CommandModule(pogiFileDirectory=self.pogiFile)

    def tearDown(self):
        self.pogiData = dict()

        """
        
        return no error code if null
        
        
        return no 
        
        
        
        """

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
    def test_