# Large test file
# No enable
# pylint: disable=too-many-lines
"""
Test camera intrinsics and extrinsics.
"""

import numpy as np
import pytest

from modules.geolocation import camera_properties


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def camera_intrinsic():
    """
    Intrinsic camera properties.
    """
    resolution_x = 2000
    resolution_y = 2000
    fov_x = np.pi / 2
    fov_y = np.pi / 2

    result, camera = camera_properties.CameraIntrinsics.create(
        resolution_x,
        resolution_y,
        fov_x,
        fov_y,
    )
    assert result
    assert camera is not None

    yield camera


class TestVectorR3Check:
    """
    Test 3D vector check.
    """

    def test_r3(self):
        """
        R^3 .
        """
        # Setup
        vec_r3 = np.empty(3)

        # Run
        result_actual = camera_properties.is_vector_r3(vec_r3)

        # Test
        assert result_actual

    def test_r2(self):
        """
        Not R^3 .
        """
        # Setup
        vec_r2 = np.empty(2)

        # Run
        result_actual = camera_properties.is_vector_r3(vec_r2)

        # Test
        assert not result_actual

    def test_matrix(self):
        """
        Matrix.
        """
        # Setup
        matrix = np.empty((2, 3))

        # Run
        result_actual = camera_properties.is_vector_r3(matrix)

        # Test
        assert not result_actual

    def test_weird_r3(self):
        """
        Weird R^3 should not pass.
        """
        # Setup
        weird_r3 = np.empty((3, 1))

        # Run
        result_actual = camera_properties.is_vector_r3(weird_r3)

        # Test
        assert not result_actual


class TestMatrixR3x3Check:
    """
    Test 3x3 matrix check.
    """

    def test_r3x3(self):
        """
        R^{3x3} .
        """
        # Setup
        matrix_r3x3 = np.empty((3, 3))

        # Run
        result_actual = camera_properties.is_matrix_r3x3(matrix_r3x3)

        # Test
        assert result_actual

    def test_r2x2(self):
        """
        Vector.
        """
        # Setup
        matrix_r2x2 = np.empty((2, 2))

        # Run
        result_actual = camera_properties.is_matrix_r3x3(matrix_r2x2)

        # Test
        assert not result_actual

    def test_r3(self):
        """
        Vector.
        """
        # Setup
        vec_r3 = np.empty(3)

        # Run
        result_actual = camera_properties.is_matrix_r3x3(vec_r3)

        # Test
        assert not result_actual

    def test_weird_r3x3(self):
        """
        Vector.
        """
        # Setup
        weird_r3x3 = np.empty((3, 3, 1))

        # Run
        result_actual = camera_properties.is_matrix_r3x3(weird_r3x3)

        # Test
        assert not result_actual


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
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
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
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
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
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
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
        Quarter turn leaning right.
        """
        # Setup
        yaw = 0.0
        pitch = 0.0
        roll = np.pi / 2

        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
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
                [1 / 2, (np.sqrt(2) - 2) / 4, (np.sqrt(2) + 2) / 4],
                [1 / 2, (np.sqrt(2) + 2) / 4, (np.sqrt(2) - 2) / 4],
                [-np.sqrt(2) / 2, 1 / 2, 1 / 2],
            ],
            dtype=np.float32,
        )

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
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
                [1 / 2, (np.sqrt(2) + 2) / 4, (-np.sqrt(2) + 2) / 4],
                [-1 / 2, (-np.sqrt(2) + 2) / 4, (np.sqrt(2) + 2) / 4],
                [np.sqrt(2) / 2, -1 / 2, 1 / 2],
            ],
            dtype=np.float32,
        )

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_bad_yaw_too_negative(self):
        """
        Expect failure.
        """
        # Setup
        yaw = -4.0
        pitch = 0.0
        roll = 0.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_yaw_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        yaw = 4.0
        pitch = 0.0
        roll = 0.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_pitch_too_negative(self):
        """
        Expect failure.
        """
        # Setup
        yaw = 0.0
        pitch = -4.0
        roll = 0.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_pitch_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        yaw = 0.0
        pitch = 4.0
        roll = 0.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_roll_too_negative(self):
        """
        Expect failure.
        """
        # Setup
        yaw = 0.0
        pitch = 0.0
        roll = -4.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_roll_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        yaw = 0.0
        pitch = 0.0
        roll = 4.0

        # Run
        result, actual = camera_properties.create_rotation_matrix_from_orientation(
            yaw,
            pitch,
            roll,
        )

        # Test
        assert not result
        assert actual is None


class TestCameraIntrinsicsCreate:
    """
    Test constructor.
    """

    def test_normal(self):
        """
        Successful construction.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = np.pi / 2
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert result
        assert actual is not None

    def test_bad_resolution_x(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = -1
        resolution_y = 2000
        fov_x = np.pi / 2
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_resolution_y(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = -1
        fov_x = np.pi / 2
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_x_negative(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = -1.0
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_x_zero(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = 0.0
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_x_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = np.pi
        fov_y = np.pi / 2

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_y_negative(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = np.pi / 2
        fov_y = -1.0

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_y_zero(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = np.pi / 2
        fov_y = 0.0

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_fov_y_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        resolution_x = 2000
        resolution_y = 2000
        fov_x = np.pi / 2
        fov_y = np.pi

        # Run
        result, actual = camera_properties.CameraIntrinsics.create(
            resolution_x,
            resolution_y,
            fov_x,
            fov_y,
        )

        # Test
        assert not result
        assert actual is None


class TestImagePixelToVector:
    """
    Test convert from image pixel to image vector.
    """

    def test_centre(self):
        """
        Centre of image.
        """
        # Setup
        pixel = 1000
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.zeros(3)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_left(self):
        """
        Left of image.
        """
        # Setup
        pixel = 0
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.array([-1.0, -1.0, -1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_right(self):
        """
        Right of image.
        """
        # Setup
        pixel = 2000
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_half_right(self):
        """
        Halfway between centre and right of image.
        """
        # Setup
        pixel = 1500
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_bad_pixel_too_positive(self):
        """
        Expect failure.
        """
        # Setup
        pixel = 2001
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_pixel_negative(self):
        """
        Expect failure.
        """
        # Setup
        pixel = -1
        resolution = 2000
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_resolution_zero(self):
        """
        Expect failure.
        """
        # Setup
        pixel = 1000
        resolution = 0
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_resolution_negative(self):
        """
        Expect failure.
        """
        # Setup
        pixel = 1000
        resolution = -1
        vec_base = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_base,
        )

        # Test
        assert not result
        assert actual is None

    def test_bad_vec_base_not_r3(self):
        """
        Expect failure.
        """
        # Setup
        pixel = 1000
        resolution = 2000
        vec_r2 = np.array([1.0, 1.0], dtype=np.float32)

        # Run
        (
            result,
            actual,
        ) = camera_properties.CameraIntrinsics._CameraIntrinsics__pixel_vector_from_image_space(  # type: ignore
            pixel,
            resolution,
            vec_r2,
        )

        # Test
        assert not result
        assert actual is None


class TestImageToCameraSpace:
    """
    Test convert from image point to camera vector.
    """

    def test_centre(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Centre of image.
        """
        # Setup
        pixel_x = 1000
        pixel_y = 1000

        expected = np.array([1.0, 0.0, 0.0])

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_top_left(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Top left corner of image.
        """
        # Setup
        pixel_x = 0
        pixel_y = 0

        expected = np.array([1.0, -1.0, -1.0])

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_bottom_right(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Bottom right corner of image.
        """
        # Setup
        pixel_x = 2000
        pixel_y = 2000

        expected = np.array([1.0, 1.0, 1.0])

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert result
        assert actual is not None
        np.testing.assert_almost_equal(actual, expected)

    def test_bad_pixel_x_negative(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Expect failure.
        """
        # Setup
        pixel_x = -1
        pixel_y = 1000

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert not result
        assert actual is None

    def test_bad_pixel_x_too_positive(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Expect failure.
        """
        # Setup
        pixel_x = 2001
        pixel_y = 1000

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert not result
        assert actual is None

    def test_bad_pixel_y_negative(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Expect failure.
        """
        # Setup
        pixel_x = 1000
        pixel_y = -1

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert not result
        assert actual is None

    def test_bad_pixel_y_too_positive(self, camera_intrinsic: camera_properties.CameraIntrinsics):
        """
        Expect failure.
        """
        # Setup
        pixel_x = 1000
        pixel_y = 2001

        # Run
        result, actual = camera_intrinsic.camera_space_from_image_space(pixel_x, pixel_y)

        # Test
        assert not result
        assert actual is None


class TestCameraExtrinsicsCreate:
    """
    Test constructor.
    """

    def test_normal(self):
        """
        Successful construction.
        """
        # Setup
        camera_position_xyz = (0.0, 0.0, 0.0)
        camera_orientation_ypr = (0.0, 0.0, 0.0)

        # Run
        result, actual = camera_properties.CameraDroneExtrinsics.create(
            camera_position_xyz,
            camera_orientation_ypr,
        )

        # Test
        assert result
        assert actual is not None

    def test_bad_orientation(self):
        """
        Expect failure.
        """
        # Setup
        camera_position_xyz = (0.0, 0.0, 0.0)
        camera_orientation_ypr = (4.0, -4.0, 4.0)

        # Run
        result, actual = camera_properties.CameraDroneExtrinsics.create(
            camera_position_xyz,
            camera_orientation_ypr,
        )

        # Test
        assert not result
        assert actual is None
