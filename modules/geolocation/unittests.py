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
    
class TestMapLocationFromPixel(unittest.TestCase):
    """
    Tests Geolocation.map_location_from_pixel()
    """
    def setUp(self):
        
        self.locator = geolocation.Geolocation()

        return
    
    def test_identity_transformation_matrix(self):

        # Setup
        identityTransformationMatrix = np.ones(shape=(3,3))
        pixelCoordinates = np.array([[2,3],[12,99],[623,126],[1604,12],[0,4]])

        expected = np.array(np.ones(shape=(5,2)))

        # Run
        actual = self.locator.map_location_from_pixel(identityTransformationMatrix, pixelCoordinates)
        
        # Test
        np.testing.assert_almost_equal(actual, expected)
    
    def test_transformation_matrix_1(self):

        # Setup
        transformationMatrix = np.array([[4,6,152],[120,5,99],[3,5,2]])
        pixelCoordinates = np.array([[2,3],[12,99],[623,126],[1604,12],[0,4]])

        expected = np.array([[7.739130435,15.39130435], 
                                [1.489681051,3.816135084], 
                                [1.359456218,30.18352659],
                                [1.362330735,39.52379975],
                                [8,5.409090909]])

        # Run
        actual = self.locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)
    
    def test_small_values_transformation_matrix(self):

        # Setup
        transformationMatrix = np.array([[0.2231,0.1222,0.0345],[0.0512,0.0041,0.0062],[0.3315,0.8720,0.1261]])
        pixelCoordinates = np.array([[2,3],[12,99],[623,126],[1604,12],[0,4]])

        expected = np.array([[0.24883274,0.03550557], 
                             [0.16376375,0.01135106], 
                             [0.48787354,0.10242681],
                             [0.66262702,0.15153561],
                             [0.14479400,0.00625329]])

        # Run
        actual = self.locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)
    
 
 
if __name__ == "__main__":
    unittest.main()