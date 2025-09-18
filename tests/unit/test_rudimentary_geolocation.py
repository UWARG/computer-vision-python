"""
Test rudimentary_geolocation.
"""

import numpy as np
import pytest

from modules import detection_in_world
from modules import detections_and_time
from modules import merged_odometry_detections
from modules.common.modules import orientation
from modules.common.modules import position_local
from modules.common.modules.logger import logger
from modules.common.modules.mavlink import drone_odometry_local
from modules.geolocation import camera_properties
from modules.geolocation import rudimentary_geolocation

FLOAT_PRECISION_TOLERANCE = 4


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name
# pylint: disable=duplicate-code


@pytest.fixture
def basic_locator() -> rudimentary_geolocation.RudimentaryGeolocation:  # type: ignore
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

    result, test_logger = logger.Logger.create("test_logger", False)
    assert result
    assert test_logger is not None

    result, locator = rudimentary_geolocation.RudimentaryGeolocation.create(
        camera_intrinsics,
        camera_extrinsics,
        test_logger,
    )
    assert result
    assert locator is not None

    yield locator  # type: ignore


@pytest.fixture
def detection_1() -> detections_and_time.Detection:  # type: ignore
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

    yield detection  # type: ignore


@pytest.fixture
def detection_2() -> detections_and_time.Detection:  # type: ignore
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

    yield detection  # type: ignore


class TestRudimentaryGeolocationRun:
    """
    Run.
    """

    def test_basic(
        self,
        basic_locator: rudimentary_geolocation.RudimentaryGeolocation,
        detection_1: detections_and_time.Detection,
        detection_2: detections_and_time.Detection,
    ) -> None:
        """
        2 detections.
        """
        # Setup
        result, drone_position = position_local.PositionLocal.create(
            0.0,
            0.0,
            -100.0,
        )
        assert result
        assert drone_position is not None

        result, drone_orientation = orientation.Orientation.create(
            0.0,
            -np.pi / 2,
            0.0,
        )
        assert result
        assert drone_orientation is not None

        result, drone_odometry = drone_odometry_local.DroneOdometryLocal.create(
            drone_position,
            drone_orientation,
        )
        assert result
        assert drone_odometry is not None

        result, merged_detections = merged_odometry_detections.MergedOdometryDetections.create(
            drone_odometry,
            [
                detection_1,
                detection_2,
            ],
        )
        assert result
        assert merged_detections is not None

        result, expected_detection_1 = detection_in_world.DetectionInWorld.create(
            # fmt: off
            np.array(
                [
                    [ 100.0, -100.0],
                    [ 100.0,  100.0],
                    [-100.0, -100.0],
                    [-100.0,  100.0],
                ],
                dtype=np.float32,
            ),
            # fmt: on
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            ),
            1,
            1.0,
        )
        assert result
        assert expected_detection_1 is not None

        result, expected_detection_2 = detection_in_world.DetectionInWorld.create(
            # fmt: off
            np.array(
                [
                    [ 100.0, -100.0],
                    [ 100.0,    0.0],
                    [   0.0, -100.0],
                    [   0.0,    0.0],
                ],
                dtype=np.float32,
            ),
            # fmt: on
            np.array(
                [50.0, -50.0],
                dtype=np.float32,
            ),
            2,
            0.5,
        )
        assert result
        assert expected_detection_2 is not None

        expected_list = [
            expected_detection_1,
            expected_detection_2,
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

    def test_bad_direction(
        self,
        basic_locator: rudimentary_geolocation.RudimentaryGeolocation,
        detection_1: detections_and_time.Detection,
    ) -> None:
        """
        Bad direction.
        """
        # Setup
        result, drone_position = position_local.PositionLocal.create(
            0.0,
            0.0,
            -100.0,
        )
        assert result
        assert drone_position is not None

        result, drone_orientation = orientation.Orientation.create(
            0.0,
            0.0,
            0.0,
        )
        assert result
        assert drone_orientation is not None

        result, drone_odometry = drone_odometry_local.DroneOdometryLocal.create(
            drone_position,
            drone_orientation,
        )
        assert result
        assert drone_odometry is not None

        result, merged_detections = merged_odometry_detections.MergedOdometryDetections.create(
            drone_odometry,
            [detection_1],
        )
        assert result
        assert merged_detections is not None

        # Run
        result, actual_list = basic_locator.run(merged_detections)

        # Test
        assert not result
        assert actual_list is None
