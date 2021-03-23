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

        
class TestOutputAnalysis(unittest.TestCase):
    """
    Tests Geolocation.outputanalysis(inputLocationTupleList)

    """
    def test_all_ones(self):
        outputAnalysis = geolocation.OutputAnalysis()

    # Setup
        outputList = np.array([[[1, 1], [1], [1]]])

        expectedCoordPair= (np.array([[ 1, 1]]))
        expectedError= (np.array([[1]]))

        # Run
        actual = outputAnalysis.get_best_location(outputList)

        np.testing.assert_equal(actual[0], expectedCoordPair)
        np.testing.assert_equal(actual[1], expectedError)

    def test_all_zeros(self):

    # Setup
        outputAnalysis = geolocation.OutputAnalysis()

        outputList = np.array([[[0, 0], [0], [0]]])

        expectedCoordPair= (np.array([[ 0, 0]]))
        expectedError= (np.array([[0]]))

        # Run
        actual = outputAnalysis.get_best_location(outputList)

        np.testing.assert_equal(actual[0], expectedCoordPair)
        np.testing.assert_equal(actual[1], expectedError)

    def test_normal_input(self):

    # Setup
        outputAnalysis = geolocation.OutputAnalysis()

        outputList = np.array([
        [[10.4, 20.5], [2.4], [0.78]],
        [[11.2, 19.9], [2.8], [0.69]],
        [[10.1, 20.3], [1.9], [0.94]]])

        expectedCoordPair = (np.array([ 10.25,  20.4 ]))
        expectedError = (np.array([ 2.36666667]))

        # Run
        actual = outputAnalysis.get_best_location(outputList)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_input_with_outlier(self):

    # Setup

        outputAnalysis = geolocation.OutputAnalysis()
        outputList = np.array([
        [[10.4, 20.5], [2.4], [0.78]],
        [[11.2, 19.9], [2.8], [0.69]],
        [[100.1, 200.3], [1.9], [0.94]]])

        expectedCoordPair = (np.array([ 10.8,  20.2]))
        expectedError = (np.array([ 2.36666667]))

        # Run
        actual = outputAnalysis.get_best_location(outputList)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)


if __name__ == "__main__":
    unittest.main()
