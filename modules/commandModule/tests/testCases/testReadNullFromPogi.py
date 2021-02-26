import json
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
    def __value_instantiate(self, key, value):
        with open(self.pogiFile, "w") as file:
            dict = {key: value}
            json.dump(dict, file, ensure_ascii=False, indent=4, sort_keys=True)

    def test_error_code_if_equals_null(self):
        with self.assertRaises(SystemExit) as cm:
            self.commandModule.set_gps_coordinates(None)
        self.assertEqual(cm.exception.code, 1)

    def test_altitude_if_equals_none(self):
        self.__value_instantiate("alt")
    def test_airspeed_if_equals_none(self):
        self.pogiData['airspeed'] = None
        with self.assertRaises(SystemExit) as cm:

        self.assertEqual


    def test_if_landed_if_equals_none(self):

    def test_euler_camera_if_equals_none(self):

    def test_euler_camera_alpha_if_equals_none(self):

    def test_euler_camera_beta_if_equals_none(self):

    def test_euler_camera_gamma_if_equals_none(self):

    def test_euler_plane_if_equals_none(self):

    def test_euler_plane_alpha_if_equals_none(self):

    def test_euler_plane_beta_if_equals_none(self):

    def test_euler_plane_gamma_if_equals_none(self):

    def test_gps_if_equals_none(self):

    def test_gps_lat_alpha_if_equals_none(self):

    def test_gps_lng_beta_if_equals_none(self):

