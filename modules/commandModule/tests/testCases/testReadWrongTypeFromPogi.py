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

    def test_error_code_if_equals_null(self):
        with self.assertRaises(SystemExit) as cm:
            self.commandModule.set_gps_coordinates(wrong_type)
        self.assertEqual(cm.exception.code, 1)

    def test_altitude_if_equals_wrong_type(self):
        with self.assertLogs(logger=self.logger, level="ERROR") as cm:
            try:
                self.commandModule.set_gps_coordinates(wrong_type)
            except SystemExit:
                pass

    def test_airspeed_if_equals_wrong_type(self):
        self.pogiData['airspeed'] = wrong_type
        with self.assertRaises(SystemExit) as cm:

        self.assertEqual

    def test_if_landed_if_equals_wrong_type(self):

    def test_euler_camera_if_equals_wrong_type(self):

    def test_euler_camera_alpha_if_equals_wrong_type(self):

    def test_euler_camera_beta_if_equals_wrong_type(self):

    def test_euler_camera_gamma_if_equals_wrong_type(self):

    def test_euler_plane_if_equals_wrong_type(self):

    def test_euler_plane_alpha_if_equals_wrong_type(self):

    def test_euler_plane_beta_if_equals_wrong_type(self):

    def test_euler_plane_gamma_if_equals_wrong_type(self):

    def test_gps_if_equals_wrong_type(self):

    def test_gps_lat_alpha_if_equals_wrong_type(self):

    def test_gps_lng_beta_if_equals_wrong_type(self):

