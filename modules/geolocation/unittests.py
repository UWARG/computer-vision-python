"""
Unit tests for geolocation module
"""

import unittest
import numpy as np

import geolocation

class TestGatherPointPairs(unittest.TestCase):
    """
    Tests Geolocation.gather_point_pairs()
    """

    def test_camera_offset_from_origin_pointing_down(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__cameraOrigin3o = np.array([2.0, 4.0, 2.0])
        locator._Geolocation__cameraDirection3c = np.array([0.0, 0.0, -1.0])
        locator._Geolocation__cameraOrientation3u = np.array([0.0, -2.0, 0.0])
        locator._Geolocation__cameraOrientation3v = np.array([-1.0, 0.0, 0.0])
        locator._Geolocation__cameraResolution = np.array([20, 10])
        locator._Geolocation__referencePixels = np.array([[0, 0],
                                                          [0, 10],
                                                          [20, 0],
                                                          [20, 10]])

        expected = np.array([[[0, 0], [4.0, 8.0]],
                             [[0, 10], [0.0, 8.0]],
                             [[20, 0], [4.0, 0.0]],
                             [[20, 10], [0.0, 0.0]]])

        # Run
        actual = locator.gather_point_pairs()

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_at_origin_pointing_slanted(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__cameraOrigin3o = np.array([0.0, 0.0, 3.0])
        locator._Geolocation__cameraDirection3c = np.array([0.0, 1.0, -1.0])
        locator._Geolocation__cameraOrientation3u = np.array([-1.0, 0.0, 0.0])
        locator._Geolocation__cameraOrientation3v = np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        locator._Geolocation__cameraResolution = np.array([10, 10])
        locator._Geolocation__referencePixels = np.array([[0, 0],
                                                          [0, 10],
                                                          [10, 0],
                                                          [10, 10]])

        expected = np.array([[[0, 0], [6 - 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                             [[0, 10], [6 + 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]],
                             [[10, 0], [-6 + 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                             [[10, 10], [-6 - 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]]])

        # Run
        actual = locator.gather_point_pairs()

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_offset_from_origin_pointing_sideways_with_some_upward_pixels(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__cameraOrigin3o = np.array([0.0, 1.0, 4.0])
        locator._Geolocation__cameraDirection3c = np.array([0.0, 1.0, 0.0])
        locator._Geolocation__cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        locator._Geolocation__cameraOrientation3v = np.array([0.0, 0.0, -2.0])
        locator._Geolocation__cameraResolution = np.array([1000, 2000])
        locator._Geolocation__referencePixels = np.array([[0, 0],  # Up
                                                          [0, 1500],
                                                          [0, 2000],
                                                          [1000, 0],  # Up
                                                          [1000, 1500],
                                                          [1000, 2000]])

        expected = np.array([[[0, 1500], [-4.0, 5.0]],
                             [[0, 2000], [-2.0, 3.0]],
                             [[1000, 1500], [4.0, 5.0]],
                             [[1000, 2000], [2.0, 3.0]]])

        # Run
        actual = locator.gather_point_pairs()

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_offset_from_origin_pointing_sideways_with_not_enough_downward_pixels(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__cameraOrigin3o = np.array([0.0, 1.0, 4.0])
        locator._Geolocation__cameraDirection3c = np.array([0.0, 1.0, 0.0])
        locator._Geolocation__cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        locator._Geolocation__cameraOrientation3v = np.array([0.0, 0.0, -2.0])
        locator._Geolocation__cameraResolution = np.array([1000, 2000])
        locator._Geolocation__referencePixels = np.array([[0, 0],  # Up
                                                          [0, 1000],  # Parallel to ground
                                                          [0, 2000],
                                                          [1000, 0],  # Up
                                                          [1000, 1000],  # Parallel to ground
                                                          [1000, 2000]])

        expected = 0

        # Run
        pairs = locator.gather_point_pairs()
        actual = np.size(pairs)

        # Test
        np.testing.assert_equal(actual, expected)


class TestPointMatrixToGeoMapping(unittest.TestCase):
    """
    Tests Geolocation.calculate_pixel_to_geo_mapping()
    """





    def test_identity_mapping(self):
        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__pixelToGeoPairs = np.array([[[1, 1], [1, 1]],
                                                          [[-1, -1], [-1, -1]],
                                                          [[1, -1], [1, -1]],
                                                          [[-1, 1], [-1, 1]]])

        expected = np.array(np.eye(3))

        # Run
        actual = locator.calculate_pixel_to_geo_mapping()

        # Test
        np.testing.assert_almost_equal(actual, expected)

        
    def test_rotation_90(self):
        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__pixelToGeoPairs = np.array([[[1, 0], [0, 1]],
                                                          [[0, 1], [-1, 0]],
                                                          [[-1, 0], [0, -1]],
                                                          [[0, -1], [1, 0]]])

        expected = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

        # Run
        actual = locator.calculate_pixel_to_geo_mapping()

        # Test
        np.testing.assert_almost_equal(actual, expected)
     
    
    def test_point_set_1(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__pixelToGeoPairs = np.array([[[0, 0], [6 - 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                                                            [[0, 10], [6 + 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]],
                                                            [[10, 0], [-6 + 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                                                            [[10, 10], [-6 - 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]]])

        expected = np.array([[((-3*np.sqrt(2))-6)/5, 0, (3*np.sqrt(2))+6],
                            [0, ((3*np.sqrt(2))+3)/5, 3],
                            [0, ((-1*np.sqrt(2))-1)/5, ((2*np.sqrt(2))+3)]])

        # Run
        actual = locator.calculate_pixel_to_geo_mapping()

        # Test
        np.testing.assert_almost_equal(actual, expected)

        
    def test_point_set_2(self):

        # Setup
        locator = geolocation.Geolocation()
        locator._Geolocation__pixelToGeoPairs = np.array([[[0, 1500], [-4.0, 5.0]],
                                                          [[0, 2000], [-2.0, 3.0]],
                                                          [[1000, 1500], [4.0, 5.0]],
                                                          [[1000, 2000], [2.0, 3.0]]])
        expected = np.array([[1/250, 0, -2],
                             [0, 1/1000, 1],
                             [0, 1/1000, -1]])
        # Run
        actual = locator.calculate_pixel_to_geo_mapping()
        # Test
        np.testing.assert_almost_equal(actual, expected)

class TestConvertInput(unittest.TestCase):

    def setUp(self):
        self.locator = geolocation.Geolocation()
        return

    def test_all_zeroes(self):

        expected_o = np.zeros(3)
        expected_c = np.array([1, 0 ,0])
        expected_u = np.array([0, 1 ,0])
        expected_v = np.array([0, 0 ,1])

        self.locator._Geolocation__latitude = 0
        self.locator._Geolocation__longitude = 0
        self.locator._Geolocation__altitude = 0
        self.locator._Geolocation__WORLD_ORIGIN = np.array([0, 0, 0])
        self.locator._Geolocation__GPS_OFFSET = np.array([0, 0, 0])
        self.locator._Geolocation__CAMERA_OFFSET = np.array([0, 0, 0])
        self.locator._Geolocation__eulerCamera = {"x": 0, "y": 0, "z": 0}
        self.locator._Geolocation__eulerPlane = {"x": 0, "y": 0, "z": 0}

        actual = self.locator.convert_input()

        expected = (expected_o, expected_c, expected_u, expected_v)
        #assert
        np.testing.assert_almost_equal(actual, expected)
        
    def test_90_degree_rotation(self):

        expected_o = np.zeros(3)
        expected_c = np.array([-1, 0, 0])
        expected_u = np.array([0, 1, 0])
        expected_v = np.array([0, 0, -1])

        self.locator._Geolocation__FOV_FACTOR = 1
        self.locator._Geolocation__latitude = 0
        self.locator._Geolocation__longitude = 0
        self.locator._Geolocation__altitude = 0
        self.locator._Geolocation__WORLD_ORIGIN = np.array([0, 0, 0])
        self.locator._Geolocation__GPS_OFFSET = np.array([0, 0, 0])
        self.locator._Geolocation__CAMERA_OFFSET = np.array([0, 0, 0])

        rightAngle = np.pi / 2
        self.locator._Geolocation__eulerPlane = {"x": rightAngle, "y": rightAngle, "z": rightAngle}
        self.locator._Geolocation__eulerCamera = {"x": rightAngle, "y": rightAngle, "z": rightAngle}

        actual = self.locator.convert_input()

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)

    def test_point_set_1(self):

        self.locator._Geolocation__latitude = -100
        self.locator._Geolocation__longitude = -100
        self.locator._Geolocation__altitude = 100
        self.locator._Geolocation__WORLD_ORIGIN = np.array([-50, 50, 50])
        self.locator._Geolocation__GPS_OFFSET = np.array([-0.5, 0, 0.5])
        self.locator._Geolocation__CAMERA_OFFSET = np.array([0.5, 0, -0.5])
        self.locator._Geolocation__eulerPlane = {"x": np.deg2rad(45), "y": np.deg2rad(180), "z": np.deg2rad(90)}
        self.locator._Geolocation__eulerCamera = {"x": np.deg2rad(90), "y": np.deg2rad(45), "z": np.deg2rad(135)}

        expected_o = np.array([-49.5 + 1/np.sqrt(2), -150.5 + (np.sqrt(2) - 2)/4, 50.5 - (np.sqrt(2) + 2)/4])
        expected_c = np.array([0.5, -0.5, -1/np.sqrt(2)])
        expected_u = np.array([(np.sqrt(2) - 2)/4, -(np.sqrt(2) + 2) / 4, 0.5])
        expected_v = np.array([-0.25 - (np.sqrt(2) + 2) / (4 * np.sqrt(2)), (2 - np.sqrt(2)) / (4 * np.sqrt(2)) - 0.25, -0.5])

        actual = self.locator.convert_input()

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)


    def test_point_set_2(self):

        self.locator._Geolocation__latitude = 50
        self.locator._Geolocation__longitude = -75
        self.locator._Geolocation__altitude = -115
        self.locator._Geolocation__WORLD_ORIGIN = np.array([-50, -25, 50])
        self.locator._Geolocation__GPS_OFFSET = np.array([-1, -1, 0])
        self.locator._Geolocation__CAMERA_OFFSET = np.array([5, 0, 2])
        self.locator._Geolocation__eulerPlane = {"x": np.deg2rad(45), "y": 0, "z": np.deg2rad(270)}
        self.locator._Geolocation__eulerCamera = {"x": np.deg2rad(180), "y": np.deg2rad(45), "z": np.deg2rad(315)}

        expected_o = np.array([-25 + 7/np.sqrt(2), 75 - 2, -165 - 5/np.sqrt(2)])
        expected_c = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
        expected_u = np.array([0, 0, -1])
        expected_v = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0])

        actual = self.locator.convert_input()

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
