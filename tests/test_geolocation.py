"""
pytests for geolocation module
"""
import pytest
import numpy as np

from modules.geolocation.geolocation import Geolocation

@pytest.fixture()
def locator():
    locator = Geolocation()
    yield locator


class TestGatherPointPairs:
    """
    Tests Geolocation.gather_point_pairs()
    """

    def test_camera_offset_from_origin_pointing_down(self, locator):

        # Setup
        cameraOrigin3o = np.array([2.0, 4.0, 2.0])
        cameraDirection3c = np.array([0.0, 0.0, -1.0])
        cameraOrientation3u = np.array([0.0, -2.0, 0.0])
        cameraOrientation3v = np.array([-1.0, 0.0, 0.0])
        referencePixels = np.array([[0, 0],
                                    [0, 10],
                                    [20, 0],
                                    [20, 10]])
        locator._Geolocation__cameraResolution = np.array([20, 10])

        expected = np.array([[[0, 0], [4.0, 8.0]],
                             [[0, 10], [0.0, 8.0]],
                             [[20, 0], [4.0, 0.0]],
                             [[20, 10], [0.0, 0.0]]])

        # Run
        actual = locator.gather_point_pairs(cameraOrigin3o, cameraDirection3c, cameraOrientation3u, cameraOrientation3v, referencePixels)

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_at_origin_pointing_slanted(self, locator):

        # Setup
        cameraOrigin3o = np.array([0.0, 0.0, 3.0])
        cameraDirection3c = np.array([0.0, 1.0, -1.0])
        cameraOrientation3u = np.array([-1.0, 0.0, 0.0])
        cameraOrientation3v = np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2])
        referencePixels = np.array([[0, 0],
                                    [0, 10],
                                    [10, 0],
                                    [10, 10]])
        locator._Geolocation__cameraResolution = np.array([10, 10])

        expected = np.array([[[0, 0], [6 - 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                             [[0, 10], [6 + 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]],
                             [[10, 0], [-6 + 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                             [[10, 10], [-6 - 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]]])

        # Run
        actual = locator.gather_point_pairs(cameraOrigin3o, cameraDirection3c, cameraOrientation3u, cameraOrientation3v, referencePixels)

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_offset_from_origin_pointing_sideways_with_some_upward_pixels(self, locator):

        # Setup
        cameraOrigin3o = np.array([0.0, 1.0, 4.0])
        cameraDirection3c = np.array([0.0, 1.0, 0.0])
        cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        cameraOrientation3v = np.array([0.0, 0.0, -2.0])
        referencePixels = np.array([[0, 0],  # Up
                                    [0, 1500],
                                    [0, 2000],
                                    [1000, 0],  # Up
                                    [1000, 1500],
                                    [1000, 2000]])
        locator._Geolocation__cameraResolution = np.array([1000, 2000])

        expected = np.array([[[0, 1500], [-4.0, 5.0]],
                             [[0, 2000], [-2.0, 3.0]],
                             [[1000, 1500], [4.0, 5.0]],
                             [[1000, 2000], [2.0, 3.0]]])

        # Run
        actual = locator.gather_point_pairs(cameraOrigin3o, cameraDirection3c, cameraOrientation3u, cameraOrientation3v, referencePixels)

        # Test
        np.testing.assert_array_almost_equal(actual, expected)


    def test_camera_offset_from_origin_pointing_sideways_with_not_enough_downward_pixels(self, locator):

        # Setup
        cameraOrigin3o = np.array([0.0, 1.0, 4.0])
        cameraDirection3c = np.array([0.0, 1.0, 0.0])
        cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        cameraOrientation3v = np.array([0.0, 0.0, -2.0])
        referencePixels = np.array([[0, 0],  # Up
                                    [0, 1000],  # Parallel to ground
                                    [0, 2000],
                                    [1000, 0],  # Up
                                    [1000, 1000],  # Parallel to ground
                                    [1000, 2000]])
        locator._Geolocation__cameraResolution = np.array([1000, 2000])

        expected = 0

        # Run
        pairs = locator.gather_point_pairs(cameraOrigin3o, cameraDirection3c, cameraOrientation3u, cameraOrientation3v, referencePixels)
        actual = np.size(pairs)

        # Test
        np.testing.assert_equal(actual, expected)


class TestGetNonCollinearPoints:
    """
    Tests get_non_collinear_points()
    """

    def test_empty_input_array(self, locator):
        # Setup
        coordinatesArray = np.array([])
        expected = 0

        # Run
        points = locator.get_non_collinear_points(coordinatesArray)
        actual = np.size(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def less_than_four_points_collinear(self, locator):
        # Setup
        coordinatesArray = np.array([[0, 0], [1, 1], [-1, -1]])
        expected = 0

        # Run
        points = locator.get_non_collinear_points(coordinatesArray)
        actual = np.size(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)


    def test_four_points_collinear(self, locator):
        coordinatesArray = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
        expected = 0

        # Run
        points = locator.get_non_collinear_points(coordinatesArray)
        actual = np.size(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_four_points_non_collinear(self, locator):
        coordinatesArray = np.array([[0, 0], [10, 12], [-1, -1], [100, 0]])
        expected = np.array([0, 1, 2, 3])

        # Run
        actual = locator.get_non_collinear_points(coordinatesArray)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_more_than_four_points_one_correct_case(self, locator):
        coordinatesArray = np.array([[0, 0], [10, 12], [-1, -1], [100, 0], [4, 4]])
        expected = np.array([0, 1, 2, 3])

        # Run
        actual = locator.get_non_collinear_points(coordinatesArray)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    
    def test_more_than_four_points_no_correct_cases(self, locator):
        coordinatesArray = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [15, 15], [3, 7]])
        expected = 0

        # Run
        points = locator.get_non_collinear_points(coordinatesArray)
        actual = np.size(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_more_than_four_points_last_correct_case(self, locator):
        coordinatesArray = np.array([[0, 0], [1, 1], [100, 0], [-1, -1], [4, 4], [-2, -2], [-4, 10]])
        expected = np.array([0, 1, 2, 6])

        # Run
        actual = locator.get_non_collinear_points(coordinatesArray)

        # Test
        np.testing.assert_almost_equal(actual, expected)


class TestPointMatrixToGeoMapping:
    """
    Tests Geolocation.calculate_pixel_to_geo_mapping()
    """

    def test_identity_mapping(self, locator):

        # Setup
        points = np.array([[[1, 1], [1, 1]],
                                                          [[-1, -1], [-1, -1]],
                                                          [[1, -1], [1, -1]],
                                                          [[-1, 1], [-1, 1]]])

        expected = np.array(np.eye(3))

        # Run
        actual = locator.calculate_pixel_to_geo_mapping(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)


    def test_rotation_90(self, locator):

        # Setup
        points = np.array([[[1, 0], [0, 1]],
                                                          [[0, 1], [-1, 0]],
                                                          [[-1, 0], [0, -1]],
                                                          [[0, -1], [1, 0]]])

        expected = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

        # Run
        actual = locator.calculate_pixel_to_geo_mapping(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)
     
    
    def test_point_set_1(self, locator):

        # Setup
        points = np.array([[[0, 0], [6 - 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                                                            [[0, 10], [6 + 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]],
                                                            [[10, 0], [-6 + 3 * np.sqrt(2), 9 - 6 * np.sqrt(2)]],
                                                            [[10, 10], [-6 - 3 * np.sqrt(2), 9 + 6 * np.sqrt(2)]]])

        expected = np.array([[((-3*np.sqrt(2))-6)/5, 0, (3*np.sqrt(2))+6],
                            [0, ((3*np.sqrt(2))+3)/5, 3],
                            [0, ((-1*np.sqrt(2))-1)/5, ((2*np.sqrt(2))+3)]])

        # Run
        actual = locator.calculate_pixel_to_geo_mapping(points)

        # Test
        np.testing.assert_almost_equal(actual, expected)

        
    def test_point_set_2(self, locator):

        # Setup
        points = np.array([[[0, 1500], [-4.0, 5.0]],
                                                          [[0, 2000], [-2.0, 3.0]],
                                                          [[1000, 1500], [4.0, 5.0]],
                                                          [[1000, 2000], [2.0, 3.0]]])
        expected = np.array([[1/250, 0, -2],
                             [0, 1/1000, 1],
                             [0, 1/1000, -1]])
        # Run
        actual = locator.calculate_pixel_to_geo_mapping(points)
        # Test
        np.testing.assert_almost_equal(actual, expected)
    
class TestMapLocationFromPixel:
    """
    Tests Geolocation.map_location_from_pixel()
    """

    def test_ones_transformation_matrix(self, locator):

        # Setup
        onesTransformationMatrix = np.ones(shape=(3,3))

        pixelCoordinates = np.array([[2,3],
                                     [12,99],
                                     [623,126],
                                     [1604,12],
                                     [0,4]])

        expected = np.ones(shape=(5,2))

        # Run
        actual = locator.map_location_from_pixel(onesTransformationMatrix, pixelCoordinates)
        
        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_set_1_int(self, locator):

        # Setup
        transformationMatrix = np.array([[4,6,152],
                                         [120,5,99],
                                         [3,5,2]])

        pixelCoordinates = np.array([[2,3],
                                     [12,99],
                                     [623,126],
                                     [1604,12],
                                     [0,4]])

        expected = np.array([[7.739130435, 15.39130435], 
                             [1.489681051, 3.816135084], 
                             [1.359456218, 30.18352659],
                             [1.362330735, 39.52379975],
                             [8,5.409090909]])

        # Run
        actual = locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)
    
    def test_set_2_float(self, locator):

        # Setup
        transformationMatrix = np.array([[66.3413,23.4231,12.8855],
                                         [95.9351,13.1522,9.5166],
                                         [3.9963,5.1629,2.6792]])

        pixelCoordinates = np.array([[61,3],
                                     [33,99],
                                     [46,26],
                                     [72,66],
                                     [94,77]])

        expected = np.array([[15.76673823, 22.52792524], 
                             [7.00192958,  6.93441577], 
                             [11.45331267, 14.85447104],
                             [10.03761573, 12.3341739],
                             [10.37866862, 12.94040829]])

        # Run
        actual = locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_small_values_transformation_matrix(self, locator):

        # Setup
        transformationMatrix = np.array([[0.2231,0.1222,0.0345],
                                         [0.0512,0.0041,0.0062],
                                         [0.3315,0.8720,0.1261]])

        pixelCoordinates = np.array([[2,3],
                                     [12,99],
                                     [623,126],
                                     [1604,12],
                                     [0,4]])

        expected = np.array([[0.24883274, 0.03550557], 
                             [0.16376375, 0.01135106], 
                             [0.48787354, 0.10242681],
                             [0.66262702, 0.15153561],
                             [0.14479400, 0.00625329]])

        # Run
        actual = locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)

    def test_homogenized_z_equals_0_case(self, locator):

        # Setup
        transformationMatrix = np.array([[5,1,5],
                                         [9,1,2],
                                         [0,0,0]])

        pixelCoordinates = np.array([[5,9],
                                     [2,5],
                                     [3,3],
                                     [11,3],
                                     [7,1]])

        expected = np.full((5,2), np.inf)

        # Run
        actual = locator.map_location_from_pixel(transformationMatrix, pixelCoordinates)

        # Test
        np.testing.assert_almost_equal(actual, expected)


class TestConvertInput:
    """
    Tests Geolocation.convert_input()
    """

    def test_all_zeroes(self, locator):

        expected_o = np.zeros(3)
        expected_c = np.array([1, 0 ,0])
        expected_u = np.array([0, 1 ,0])
        expected_v = np.array([0, 0 ,1])

        locator._Geolocation__latitude = 0
        locator._Geolocation__longitude = 0
        locator._Geolocation__altitude = 0
        locator._Geolocation__WORLD_ORIGIN = np.array([0, 0, 0])
        locator._Geolocation__GPS_OFFSET = np.array([0, 0, 0])
        locator._Geolocation__CAMERA_OFFSET = np.array([0, 0, 0])
        eulerPlane = {"roll": 0, "pitch": 0, "yaw": 0}
        eulerCamera = {"roll": 0, "pitch": 0, "yaw": 0}

        actual = locator.convert_input(eulerPlane, eulerCamera)

        expected = (expected_o, expected_c, expected_u, expected_v)
        #assert
        np.testing.assert_almost_equal(actual, expected)

    def test_90_degree_rotation(self, locator):

        expected_o = np.zeros(3)
        expected_c = np.array([-1, 0, 0])
        expected_u = np.array([0, 2, 0])
        expected_v = np.array([0, 0, -2])

        locator._Geolocation__FOV_FACTOR_H = 2
        locator._Geolocation__FOV_FACTOR_V = 2
        locator._Geolocation__latitude = 0
        locator._Geolocation__longitude = 0
        locator._Geolocation__altitude = 0
        locator._Geolocation__WORLD_ORIGIN = np.array([0, 0, 0])
        locator._Geolocation__GPS_OFFSET = np.array([0, 0, 0])
        locator._Geolocation__CAMERA_OFFSET = np.array([0, 0, 0])

        rightAngle = np.pi / 2
        eulerPlane = {"roll": rightAngle, "pitch": rightAngle, "yaw": rightAngle}
        eulerCamera = {"roll": rightAngle, "pitch": rightAngle, "yaw": rightAngle}

        actual = locator.convert_input(eulerPlane, eulerCamera)

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)

    def test_point_set_1(self, locator):

        locator._Geolocation__latitude = -100
        locator._Geolocation__longitude = -100
        locator._Geolocation__altitude = 100
        locator._Geolocation__WORLD_ORIGIN = np.array([-50, 50, 50])
        locator._Geolocation__GPS_OFFSET = np.array([-0.5, 0, -0.5])
        locator._Geolocation__CAMERA_OFFSET = np.array([0.5, 0, 0.5])
        eulerPlane = {"roll": np.deg2rad(45), "pitch": np.deg2rad(180), "yaw": np.deg2rad(90)}
        eulerCamera = {"roll": np.deg2rad(90), "pitch": np.deg2rad(45), "yaw": np.deg2rad(135)}

        expected_o = np.array([-(50 + np.sqrt(2) /2), -(150 + np.sqrt(2) /2), 49])
        expected_c = np.array([(-np.sqrt(2) - 2) / 4, 0.5, (-np.sqrt(2) + 2) / 4])
        expected_u = np.array([(-np.sqrt(2) + 2) / 4, 0.5, (-np.sqrt(2) - 2) / 4])
        expected_v = np.array([-0.5, -np.sqrt(2) / 2, -0.5])

        actual = locator.convert_input(eulerPlane, eulerCamera)

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)


    def test_point_set_2(self, locator):

        locator._Geolocation__longitude = -50
        locator._Geolocation__latitude = 75
        locator._Geolocation__altitude = 115
        locator._Geolocation__WORLD_ORIGIN = np.array([50, 25, -50])
        locator._Geolocation__GPS_OFFSET = np.array([-1, -1, 0])
        locator._Geolocation__CAMERA_OFFSET = np.array([5, 0, 2])
        eulerPlane = {"roll": np.deg2rad(45), "pitch": 0, "yaw": np.deg2rad(270)}
        eulerCamera = {"roll": np.deg2rad(180), "pitch": np.deg2rad(45), "yaw": np.deg2rad(315)}

        expected_o = np.array([(-98.5 - (np.sqrt(2) * 2)), (46 + np.sqrt(2) / 2), (167.5 + 2 * np.sqrt(2))])
        expected_c = np.array([(-np.sqrt(2) + 2) / 4, -.5, (-np.sqrt(2) - 2) / 4])
        expected_u = np.array([-0.5, (np.sqrt(2)) / 2, -.5])
        expected_v = np.array([(np.sqrt(2) + 2) / 4, .5, (np.sqrt(2) - 2) / 4])

        actual = locator.convert_input(eulerPlane, eulerCamera)

        expected = (expected_o, expected_c, expected_u, expected_v)
        # assert
        np.testing.assert_almost_equal(actual, expected)



class TestGetBestLocation():
    """
    Tests Geolocation.locator(inputLocationTupleList)
    """

    def test_all_ones(self, locator):

        # Setup
        listToAnalyze = np.array([[[1, 1], [1], [1]]], dtype=object)

        expectedCoordPair = np.array([[1,1]])
        expectedError = np.array([[1]])

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_all_zeros(self, locator):

        # Setup
        listToAnalyze = np.array([[[0,0],[0],[0]]], dtype=object)

        expectedCoordPair = np.array([[0,0]])
        expectedError = np.array([[0]])

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_equal(actual[0], expectedCoordPair)
        np.testing.assert_equal(actual[1], expectedError)

    def test_normal_input(self, locator):

        # Setup
        listToAnalyze = np.array([
                                    [[10, 20], [2.3], [0.5]],
                                    [[11, 19], [2.2], [0.7]],
                                    [[12, 21], [2.1], [0.9]]
                                 ], dtype=object)

        expectedCoordPair = (np.array([[11],[20]]))
        expectedError = (np.array([2.2]))

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_points_with_one_outlier(self, locator):

        # Setup
        listToAnalyze = np.array([
                                    [[10, 20], [2.3], [0.5]],
                                    [[11, 19], [2.2], [0.7]],
                                    [[120, 180], [2.1], [0.9]]
                                 ], dtype=object)

        expectedCoordPair = (np.array([[10.5],[19.5]]))
        expectedError = (np.array([2.2]))

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_all_points_too_far_from_each_other(self, locator):

        # Setup
        listToAnalyze = np.array([
                                    [[10, 20], [2.4], [0.78]],
                                    [[300, 400], [2.8], [0.7]],
                                    [[100, 200], [3.4], [0.94]]
                                 ], dtype=object)

        expectedCoordPair = (np.array([[100],[200]]))
        expectedError = (np.array([2.8666666666667]))

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_if_one_error_is_outlier(self, locator):

        # Setup
        listToAnalyze = np.array([
                                    [[10, 20], [2.3], [0.5]],
                                    [[11, 19], [2.1], [0.7]],
                                    [[14, 23], [2.4], [0.7]],
                                    [[12, 21], [15.0], [0.9]]
                                 ], dtype=object)

        expectedCoordPair = (np.array([[11.75],[20.75]]))
        expectedError = (np.array([2.2666666666667]))

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)

    def test_all_errors_too_far_from_each_other(self, locator):

        # Setup
        listToAnalyze = np.array([
                                    [[10, 20], [2.0], [0.5]],
                                    [[11, 19], [65.0], [0.7]],
                                    [[12, 21], [15.0], [0.9]]
                                 ], dtype=object)

        expectedCoordPair = (np.array([[11],[20]]))
        expectedError = (np.array([15.0]))

        # Run
        actual = locator.get_best_location(listToAnalyze)

        np.testing.assert_almost_equal(actual[0], expectedCoordPair)
        np.testing.assert_almost_equal(actual[1], expectedError)


class TestLatLonConversion:

    def test_invertible_zeroes(self, locator):

        # Setup
        locator._Geolocation__LAT_ORIGIN = 43.43
        locator._Geolocation__LON_ORIGIN = -80.58
        expectedX = 0
        expectedY = 0

        # Run
        lat, lon = locator.lat_lon_from_local(expectedX, expectedY)
        actualX, actualY = locator.local_from_lat_lon(lat, lon)

        # Test
        np.testing.assert_almost_equal(actualX, expectedX)
        np.testing.assert_almost_equal(actualY, expectedY)

    def test_invertible_positive(self, locator):

        # Setup
        locator._Geolocation__LAT_ORIGIN = 43.43
        locator._Geolocation__LON_ORIGIN = -80.58
        expectedX = 500
        expectedY = 360

        # Run
        lat, lon = locator.lat_lon_from_local(expectedX, expectedY)
        actualX, actualY = locator.local_from_lat_lon(lat, lon)

        # Test
        np.testing.assert_almost_equal(actualX, expectedX)
        np.testing.assert_almost_equal(actualY, expectedY)

    def test_invertible_negative(self, locator):

        # Setup
        locator._Geolocation__LAT_ORIGIN = 43.43
        locator._Geolocation__LON_ORIGIN = -80.58
        expectedX = -128
        expectedY = -1024

        # Run
        lat, lon = locator.lat_lon_from_local(expectedX, expectedY)
        actualX, actualY = locator.local_from_lat_lon(lat, lon)

        # Test
        np.testing.assert_almost_equal(actualX, expectedX)
        np.testing.assert_almost_equal(actualY, expectedY)

    def test_invertible_positiveX_negativeY(self, locator):

        # Setup
        locator._Geolocation__LAT_ORIGIN = 43.43
        locator._Geolocation__LON_ORIGIN = -80.58
        expectedX = 600
        expectedY = -900

        # Run
        lat, lon = locator.lat_lon_from_local(expectedX, expectedY)
        actualX, actualY = locator.local_from_lat_lon(lat, lon)

        # Test
        np.testing.assert_almost_equal(actualX, expectedX)
        np.testing.assert_almost_equal(actualY, expectedY)

    def test_invertible_negativeX_positiveY(self, locator):

        # Setup
        locator._Geolocation__LAT_ORIGIN = 43.43
        locator._Geolocation__LON_ORIGIN = -80.58
        expectedX = -440
        expectedY = 69

        # Run
        lat, lon = locator.lat_lon_from_local(expectedX, expectedY)
        actualX, actualY = locator.local_from_lat_lon(lat, lon)

        # Test
        np.testing.assert_almost_equal(actualX, expectedX)
        np.testing.assert_almost_equal(actualY, expectedY)
