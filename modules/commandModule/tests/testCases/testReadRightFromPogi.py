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

    def test_error_code_if_correct(self):
        with self.assertRaises(SystemExit) as cm:
            self.commandModule.set_gps_coordinates(None)
        self.assertEqual(cm.exception.code, 1)

    def test_altitude_if_correct(self):
        with self.assertLogs(logger=self.logger, level="ERROR") as cm:
            try:
                self.commandModule.set_gps_coordinates(None)
            except SystemExit:
                pass

    def test_airspeed_if_correct(self):
        self.pogiData['airspeed'] = None
        with self.assertRaises(SystemExit) as cm:

        self.assertEqual

    def test_if_landed_if_correct(self):

    def test_euler_camera_if_correct(self):

    def test_euler_plane_if_correct(self):

    def test_gps_if_correct(self):


