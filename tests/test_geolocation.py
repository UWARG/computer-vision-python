"""
Test geolocation.
"""

import numpy as np
import pytest

from modules.geolocation import geolocation


class TestRotationMatrix:
    """
    Test rotation matrix.
    """
    def test_no_rotation(self):
        """
        Identity.
        """
        # Setup
        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        expected = np.identity(3, dtype=np.float32)

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_allclose(actual, expected)

    def test_yaw_quarter(self):
        """
        Quarter turn towards east from north.
        """
        # Setup
        yaw = np.pi / 2
        pitch = 0.0
        roll = 0.0

        expected = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
            ],
            dtype=np.float32
        )

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_pitch_quarter(self):
        """
        Quarter turn towards up from forward.
        """
        # Setup
        yaw = 0.0
        pitch = np.pi / 2
        roll = 0.0

        expected = np.array(
            [
                [ 0.0, 0.0, 1.0],
                [ 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32
        )

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_roll_quarter(self):
        """
        Quarter turn towards the right.
        """
        # Setup
        yaw = 0.0
        pitch = 0.0
        roll = np.pi / 2

        expected = np.array(
            [
                [1.0, 0.0,  0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0,  0.0],
            ],
            dtype=np.float32
        )

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_combined_rotations_positive(self):
        """
        Each in 45° positive direction.
        """
        # Setup
        yaw = np.pi / 4
        pitch = np.pi / 4
        roll = np.pi / 4

        expected = np.array(
            [
                [          1 / 2, (np.sqrt(2) - 2) / 4, (np.sqrt(2) + 2) / 4],
                [          1 / 2, (np.sqrt(2) + 2) / 4, (np.sqrt(2) - 2) / 4],
                [-np.sqrt(2) / 2,                1 / 2,                1 / 2],
            ],
            dtype=np.float32
        )

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_combined_rotations_negative(self):
        """
        Each in 45° negative direction.
        """
        # Setup
        yaw = -np.pi / 4
        pitch = -np.pi / 4
        roll = -np.pi / 4

        expected = np.array(
            [
                [         1 / 2,  (np.sqrt(2) + 2) / 4, (-np.sqrt(2) + 2) / 4],
                [        -1 / 2, (-np.sqrt(2) + 2) / 4,  (np.sqrt(2) + 2) / 4],
                [np.sqrt(2) / 2,                -1 / 2,                 1 / 2],
            ],
            dtype=np.float32
        )

        # Run
        result, actual = geolocation.Geolocation.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)


# TODO: Many other tests
