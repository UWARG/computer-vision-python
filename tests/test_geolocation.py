"""
Test geolocation.
"""

import numpy as np
import pytest

from modules import detection_in_world
from modules import detections_and_time
from modules.geolocation import camera_properties
from modules.geolocation import geolocation


FLOAT_PRECISION_TOLERANCE = 4


@pytest.fixture
def basic_locator():
    """
    Forwards pointing camera.
    """
    result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
        2000,
        2000,
        np.pi / 2,
        np.pi / 2,
    )
    assert result
    assert camera_intrinsics is not None

    result, camera_extrinsics = camera_properties.CameraDroneExtrinsics.create(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    assert result
    assert camera_extrinsics is not None

    result, locator = geolocation.Geolocation.create(camera_intrinsics, camera_extrinsics)
    assert result
    assert locator is not None

    yield locator


@pytest.fixture
def intermediate_locator():
    """
    Downwards pointing camera offset towards front of drone.
    """
    result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
        2000,
        2000,
        np.pi / 2,
        np.pi / 2,
    )
    assert result
    assert camera_intrinsics is not None

    result, camera_extrinsics = camera_properties.CameraDroneExtrinsics.create(
        (1.0, 0.0, 0.0),
        (0.0, -np.pi / 2, 0.0),
    )
    assert result
    assert camera_extrinsics is not None

    result, locator = geolocation.Geolocation.create(camera_intrinsics, camera_extrinsics)
    assert result
    assert locator is not None

    yield locator


@pytest.fixture
def advanced_locator():
    """
    Camera angled at 75° upward.
    Drone is expected to rotate it downwards.
    """
    result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
        2000,
        2000,
        np.pi / 3,
        np.pi / 3,
    )
    assert result
    assert camera_intrinsics is not None

    result, camera_extrinsics = camera_properties.CameraDroneExtrinsics.create(
        (1.0, 0.0, 0.0),
        (0.0, np.pi * 5 / 12, np.pi / 2),
    )
    assert result
    assert camera_extrinsics is not None

    result, locator = geolocation.Geolocation.create(camera_intrinsics, camera_extrinsics)
    assert result
    assert locator is not None

    yield locator


@pytest.fixture
def detection_bottom_right_point():
    """
    Bounding box is a single point.
    """
    result, detection = detections_and_time.Detection.create(
        np.array([2000.0, 2000.0, 2000.0, 2000.0], dtype=np.float32),
        0,
        0.01,
    )
    assert result
    assert detection is not None

    yield detection


@pytest.fixture
def detection_centre_left_point():
    """
    Bounding box is a single point.
    """
    result, detection = detections_and_time.Detection.create(
        np.array([0.0, 1000.0, 0.0, 1000.0], dtype=np.float32),
        0,
        0.1,
    )
    assert result
    assert detection is not None

    yield detection


@pytest.fixture
def detection1():
    """
    Entire image.
    """
    result, detection = detections_and_time.Detection.create(
        np.array([0.0, 0.0, 2000.0, 2000.0], dtype=np.float32),
        1,
        1.0 / 1,
    )
    assert result
    assert detection is not None

    yield detection


@pytest.fixture
def detection2():
    """
    Quadrant.
    """
    result, detection = detections_and_time.Detection.create(
        np.array([0.0, 0.0, 1000.0, 1000.0], dtype=np.float32),
        2,
        1.0 / 2,
    )
    assert result
    assert detection is not None

    yield detection


@pytest.fixture
def affine_matrix():
    """
    3x3 homogeneous.
    """
    matrix = np.array(
        [
            [0.0, 1.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 0.0,  1.0],
        ],
        dtype=np.float32,
    )

    yield matrix

@pytest.fixture
def non_affine_matrix():
    """
    3x3 homogeneous.
    """
    matrix = np.array(
        [
            [0.0, 1.0, -1.0],
            [2.0, 0.0, -1.0],
            [1.0, 1.0,  1.0],
        ],
        dtype=np.float32,
    )

    yield matrix


class TestGeolocationCreate:
    """
    Test constructor.
    """
    def test_normal(self):
        """
        Successful construction.
        """
        result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
            2000,
            2000,
            np.pi / 2,
            np.pi / 2,
        )
        assert result
        assert camera_intrinsics is not None

        result, camera_extrinsics = camera_properties.CameraDroneExtrinsics.create(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        assert result
        assert camera_extrinsics is not None

        result, locator = geolocation.Geolocation.create(camera_intrinsics, camera_extrinsics)
        assert result
        assert locator is not None


class TestGroundIntersection:
    """
    Test where vector intersects with ground.
    """
    def test_above_origin_directly_down(self):
        """
        Above origin, directly down.
        """
        # Setup
        vec_camera_in_world_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        vec_down = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        expected = np.array([0.0, 0.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_down,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected, decimal=FLOAT_PRECISION_TOLERANCE)

    def test_non_origin_directly_down(self):
        """
        Directly down.
        """
        # Setup
        vec_camera_in_world_position = np.array([100.0, -100.0, -100.0], dtype=np.float32)
        vec_down = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        expected = np.array([100.0, -100.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_down,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected, decimal=FLOAT_PRECISION_TOLERANCE)

    def test_above_origin_angled_down(self):
        """
        Above origin, angled down towards positive.
        """
        # Setup
        vec_camera_in_world_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        vec_down = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.array([100.0, 100.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_down,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected, decimal=FLOAT_PRECISION_TOLERANCE)

    def test_non_origin_angled_down(self):
        """
        Angled down towards origin.
        """
        # Setup
        vec_camera_in_world_position = np.array([100.0, -100.0, -100.0], dtype=np.float32)
        vec_down = np.array([-1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.array([0.0, 0.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_down,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected, decimal=FLOAT_PRECISION_TOLERANCE)

    def test_bad_almost_horizontal(self):
        """
        False, None .
        """
        # Setup
        vec_camera_in_world_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        vec_horizontal = np.array([10.0, 0.0, 1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_horizontal,
            )
        # pylint: enable=protected-access

        # Test
        assert not result
        assert actual is None

    def test_bad_upwards(self):
        """
        False, None .
        """
        # Setup
        vec_camera_in_world_position = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        vec_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_camera_in_world_position,
                vec_up,
            )
        # pylint: enable=protected-access

        # Test
        assert not result
        assert actual is None

    def test_bad_underground(self):
        """
        False, None .
        """
        # Setup
        vec_underground = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vec_down = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__ground_intersection_from_vector(  # type: ignore
                vec_underground,
                vec_down,
            )
        # pylint: enable=protected-access

        # Test
        assert not result
        assert actual is None


class TestPerspectiveTransformMatrix:
    """
    Test perspective transform creation.
    """
    def test_basic_above_origin_pointed_down(self, basic_locator: geolocation.Geolocation):
        """
        Above origin, directly down.
        """
        # Setup
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            0.0,
            -np.pi / 2,
            0.0,
        )
        assert result
        assert drone_rotation_matrix is not None

        drone_position_ned = np.array([0.0, 0.0, -100.0], dtype=np.float32)

        vec_ground_expected = np.array([-100.0, 100.0, 1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            basic_locator._Geolocation__get_perspective_transform_matrix(  # type: ignore
                drone_rotation_matrix,
                drone_position_ned,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        vec_ground = actual @ np.array([2000, 2000, 1])
        vec_ground_normalized = vec_ground / vec_ground[2]
        np.testing.assert_almost_equal(
            vec_ground_normalized,
            vec_ground_expected,
            decimal=FLOAT_PRECISION_TOLERANCE,
        )

    def test_intermediate_above_origin_pointing_north(self,
                                                      intermediate_locator: geolocation.Geolocation):
        """
        Positioned so that the camera is above the origin directly down (but the drone is not).
        """
        # Setup
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            0.0,
            0.0,
            0.0,
        )
        assert result
        assert drone_rotation_matrix is not None

        drone_position_ned = np.array([-1.0, 0.0, -100.0], dtype=np.float32)

        vec_ground_expected = np.array([-100.0, 100.0, 1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            intermediate_locator._Geolocation__get_perspective_transform_matrix(  # type: ignore
                drone_rotation_matrix,
                drone_position_ned,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        vec_ground = actual @ np.array([2000, 2000, 1])
        vec_ground_normalized = vec_ground / vec_ground[2]
        np.testing.assert_almost_equal(
            vec_ground_normalized,
            vec_ground_expected,
            decimal=FLOAT_PRECISION_TOLERANCE,
        )

    def test_intermediate_above_origin_pointing_west(self,
                                                     intermediate_locator: geolocation.Geolocation):
        """
        Positioned so that the camera is above the origin directly down (but the drone is not).
        """
        # Setup
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            -np.pi / 2,
            0.0,
            0.0,
        )
        assert result
        assert drone_rotation_matrix is not None

        drone_position_ned = np.array([0.0, 1.0, -100.0], dtype=np.float32)

        vec_ground_expected = np.array([100.0, 100.0, 1.0], dtype=np.float32)

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            intermediate_locator._Geolocation__get_perspective_transform_matrix(  # type: ignore
                drone_rotation_matrix,
                drone_position_ned,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        vec_ground = actual @ np.array([2000, 2000, 1])
        vec_ground_normalized = vec_ground / vec_ground[2]
        np.testing.assert_almost_equal(
            vec_ground_normalized,
            vec_ground_expected,
            decimal=FLOAT_PRECISION_TOLERANCE,
        )

    def test_advanced(self, advanced_locator: geolocation.Geolocation):
        """
        Camera is north of origin with an angle from vertical. Also rotated.
        """
        # Setup
        # TODO
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            0.0,
            np.pi / 12,
            -np.pi,
        )
        assert result
        assert drone_rotation_matrix is not None

        drone_position_ned = np.array(
            [
                10.0 - np.cos(-np.pi / 12),  # Camera at 10 units forward
                0.0,
                -100.0 - np.sin(-np.pi / 12),  # Camera at 100 units above ground
            ],
            dtype=np.float32,
        )

        vec_ground_sanity_expected = np.array([10.0, 0.0, 1.0], dtype=np.float32)
        vec_ground_expected = np.array(
            [
                10.0 + 100.0 * np.sqrt(3),
                100.0,
                1.0,
            ],
            dtype=np.float32,
        )

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            advanced_locator._Geolocation__get_perspective_transform_matrix(  # type: ignore
                drone_rotation_matrix,
                drone_position_ned,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        vec_ground_sanity = actual @ np.array([0, 1000, 1])
        vec_ground_sanity_actual = vec_ground_sanity / vec_ground_sanity[2]
        np.testing.assert_almost_equal(
            vec_ground_sanity_actual,
            vec_ground_sanity_expected,
            decimal=FLOAT_PRECISION_TOLERANCE,
        )

        vec_ground = actual @ np.array([2000, 2000, 1])
        vec_ground_normalized = vec_ground / vec_ground[2]
        np.testing.assert_almost_equal(
            vec_ground_normalized,
            vec_ground_expected,
            decimal=FLOAT_PRECISION_TOLERANCE,
        )

    def test_bad_direction(self, basic_locator: geolocation.Geolocation):
        """
        Camera pointing forward.
        """
        # Setup
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            0.0,
            0.0,
            0.0,
        )
        assert result
        assert drone_rotation_matrix is not None

        drone_position_ned = np.array(
            [
                0.0,
                0.0,
                -100.0,
            ],
            dtype=np.float32,
        )

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            basic_locator._Geolocation__get_perspective_transform_matrix(  # type: ignore
                drone_rotation_matrix,
                drone_position_ned,
            )
        # pylint: enable=protected-access

        # Test
        assert not result
        assert actual is None


class TestGeolocationConvertDetection:
    """
    Test extract and convert.
    """
    def test_normal1(self, detection1: detections_and_time.Detection, affine_matrix: np.ndarray):
        """
        Normal detection and matrix.
        """
        # Setup
        result, expected = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [  -1.0,   -1.0],
                    [  -1.0, 3999.0],
                    [1999.0,   -1.0],
                    [1999.0, 3999.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [999.0, 1999.0],
                dtype=np.float32,
            ),
            1,
            1.0,
        )
        assert result
        assert expected is not None

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__convert_detection_to_world_from_image(  # type: ignore
                detection1,
                affine_matrix,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        np.testing.assert_almost_equal(actual.vertices, expected.vertices)
        np.testing.assert_almost_equal(actual.centre, expected.centre)
        assert actual.label == expected.label
        np.testing.assert_almost_equal(actual.confidence, expected.confidence)

    def test_normal2(self, detection2: detections_and_time.Detection, affine_matrix: np.ndarray):
        """
        Normal detection and matrix.
        """
        # Setup
        result, expected = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [ -1.0,   -1.0],
                    [ -1.0, 1999.0],
                    [999.0,   -1.0],
                    [999.0, 1999.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [499.0, 999.0],
                dtype=np.float32,
            ),
            2,
            0.5,
        )
        assert result
        assert expected is not None

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__convert_detection_to_world_from_image(  # type: ignore
                detection2,
                affine_matrix,
            )
        # pylint: enable=protected-access

        # Test
        assert result
        assert actual is not None

        np.testing.assert_almost_equal(actual.vertices, expected.vertices)
        np.testing.assert_almost_equal(actual.centre, expected.centre)
        assert actual.label == expected.label
        np.testing.assert_almost_equal(actual.confidence, expected.confidence)

    def test_bad_not_homogeneous(self, detection1: detections_and_time.Detection, non_affine_matrix: np.ndarray):
        """
        Nonhomogeneous matrix.
        """
        # Setup

        # Run
        # Access required for test
        # pylint: disable=protected-access
        result, actual = \
            geolocation.Geolocation._Geolocation__convert_detection_to_world_from_image(  # type: ignore
                detection1,
                non_affine_matrix,
            )
        # pylint: enable=protected-access

        # Test
        assert not result
        assert actual is None


class TestGeolocationRun:
    """
    Run.
    """
    def test_basic(self,
                       basic_locator: geolocation.Geolocation,
                       detection1: detections_and_time.Detection,
                       detection2: detections_and_time.Detection):
        """
        2 detections.
        """
        # Setup
        result, drone_position = \
            geolocation.merged_odometry_detections.odometry_and_time.DronePosition.create(
                0.0,
                0.0,
                100.0,
            )
        assert result
        assert drone_position is not None

        result, drone_orientation = \
            geolocation.merged_odometry_detections.odometry_and_time.DroneOrientation.create(
                0.0,
                -np.pi / 2,
                0.0,
            )
        assert result
        assert drone_orientation is not None

        merged_detections = geolocation.merged_odometry_detections.MergedOdometryDetections(
            drone_position,
            drone_orientation,
            [
                detection1,
                detection2,
            ],
        )

        result, expected_detection1 = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [ 100.0, -100.0],
                    [ 100.0,  100.0],
                    [-100.0, -100.0],
                    [-100.0,  100.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            ),
            1,
            1.0,
        )
        assert result
        assert expected_detection1 is not None

        result, expected_detection2 = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [ 100.0, -100.0],
                    [ 100.0,    0.0],
                    [   0.0, -100.0],
                    [   0.0,    0.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [50.0, -50.0],
                dtype=np.float32,
            ),
            2,
            0.5,
        )
        assert result
        assert expected_detection2 is not None

        expected_list = [
            expected_detection1,
            expected_detection2,
        ]

        # Run
        result, actual_list = basic_locator.run(merged_detections)

        # Test
        assert result
        assert actual_list is not None

        assert len(actual_list) == len(expected_list)
        for i, actual in enumerate(actual_list):
            np.testing.assert_almost_equal(actual.vertices, expected_list[i].vertices)
            np.testing.assert_almost_equal(actual.centre, expected_list[i].centre)
            assert actual.label == expected_list[i].label
            np.testing.assert_almost_equal(actual.confidence, expected_list[i].confidence)

    def test_advanced(self,
                          advanced_locator: geolocation.Geolocation,
                          detection_bottom_right_point: detections_and_time.Detection,
                          detection_centre_left_point: detections_and_time.Detection):
        # Setup
        result, drone_position = \
            geolocation.merged_odometry_detections.odometry_and_time.DronePosition.create(
                10.0 - np.cos(-np.pi / 12),  # Camera at 10 units forward
                0.0,
                100.0 + np.sin(-np.pi / 12),  # Camera at 100 units above ground
            )
        assert result
        assert drone_position is not None

        result, drone_orientation = \
            geolocation.merged_odometry_detections.odometry_and_time.DroneOrientation.create(
                0.0,
                np.pi / 12,
                -np.pi,
            )
        assert result
        assert drone_orientation is not None

        merged_detections = geolocation.merged_odometry_detections.MergedOdometryDetections(
            drone_position,
            drone_orientation,
            [
                detection_bottom_right_point,
                detection_centre_left_point,
            ],
        )

        result, expected_bottom_right = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [10.0 + 100.0 * np.sqrt(3), 100.0],
                    [10.0 + 100.0 * np.sqrt(3), 100.0],
                    [10.0 + 100.0 * np.sqrt(3), 100.0],
                    [10.0 + 100.0 * np.sqrt(3), 100.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [10.0 + 100.0 * np.sqrt(3), 100.0],
                dtype=np.float32,
            ),
            0,
            0.01,
        )
        assert result
        assert expected_bottom_right is not None

        result, expected_centre_left = detection_in_world.DetectionInWorld.create(
            np.array(
                [
                    [10.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 0.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [10.0, 0.0],
                dtype=np.float32,
            ),
            0,
            0.1,
        )
        assert result
        assert expected_centre_left is not None

        expected_list = [
            expected_bottom_right,
            expected_centre_left,
        ]

        # Run
        result, actual_list = advanced_locator.run(merged_detections)

        # Test
        assert result
        assert actual_list is not None

        assert len(actual_list) == len(expected_list)
        for i, actual in enumerate(actual_list):
            np.testing.assert_almost_equal(actual.vertices, expected_list[i].vertices, FLOAT_PRECISION_TOLERANCE)
            np.testing.assert_almost_equal(actual.centre, expected_list[i].centre, FLOAT_PRECISION_TOLERANCE)
            assert actual.label == expected_list[i].label
            np.testing.assert_almost_equal(actual.confidence, expected_list[i].confidence)

    def test_bad_direction(self,
                               basic_locator: geolocation.Geolocation,
                               detection1: detections_and_time.Detection):
        """
        Bad direction.
        """
        # Setup
        result, drone_position = \
            geolocation.merged_odometry_detections.odometry_and_time.DronePosition.create(
                0.0,
                0.0,
                100.0,
            )
        assert result
        assert drone_position is not None

        result, drone_orientation = \
            geolocation.merged_odometry_detections.odometry_and_time.DroneOrientation.create(
                0.0,
                0.0,
                0.0,
            )
        assert result
        assert drone_orientation is not None

        merged_detections = geolocation.merged_odometry_detections.MergedOdometryDetections(
            drone_position,
            drone_orientation,
            [detection1],
        )

        # Run
        result, actual_list = basic_locator.run(merged_detections)

        # Test
        assert not result
        assert actual_list is None
