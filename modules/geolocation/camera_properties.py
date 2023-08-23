"""
Camera intrinsic and extrinsic properties.
"""

import numpy as np


def is_vector_r3(vec: np.ndarray) -> bool:
    """
    Checks if the numpy array is a vector in R^3 .
    """
    return vec.shape == (3,)


def is_matrix_r3x3(matrix: np.ndarray) -> bool:
    """
    Checks if the numpy array is a matrix in R^{3x3} .
    """
    return matrix.shape == (3, 3)


def create_rotation_matrix_from_orientation(yaw: float,
                                            pitch: float,
                                            roll: float) -> "tuple[bool, np.ndarray | None]":
    """
    Creates a rotation matrix from yaw pitch roll in NED system (x forward, y right, z down).
    Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order.
    See: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations

    yaw, pitch, roll are in radians from -pi to pi .
    """
    if yaw < -np.pi or yaw > np.pi:
        return False, None

    if pitch < -np.pi or pitch > np.pi:
        return False, None

    if roll < -np.pi or roll > np.pi:
        return False, None

    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [        0.0,          0.0, 1.0],
        ],
        dtype=np.float32,
    )

    pitch_matrix = np.array(
        [
            [ np.cos(pitch), 0.0, np.sin(pitch)],
            [           0.0, 1.0,           0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float32,
    )

    roll_matrix = np.array(
        [
            [1.0,          0.0,           0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll),  np.cos(roll)],
        ],
        dtype=np.float32,
    )

    rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix

    return True, rotation_matrix


class CameraIntrinsics:
    """
    Camera space information.

    Image follows xy system with top-left as origin, pixel coordinates
    refer to their top-left corner.
    """
    __create_key = object()

    @classmethod
    def create(cls,
               resolution_x: int,
               resolution_y: int,
               fov_x: float,
               fov_y: float) -> "tuple[bool, CameraIntrinsics | None]":
        """
        Resolution is in pixels.
        Field of view in radians horizontally and vertically across the image (edge to edge).
        """
        if resolution_x < 0:
            return False, None

        if resolution_y < 0:
            return False, None

        if fov_x <= 0.0 or fov_x >= np.pi:
            return False, None

        if fov_y <= 0.0 or fov_y >= np.pi:
            return False, None

        u_scalar = np.tan(fov_x / 2)
        if not np.isfinite(u_scalar):
            return False, None

        v_scalar = np.tan(fov_y / 2)
        if not np.isfinite(v_scalar):
            return False, None

        return True, CameraIntrinsics(
            cls.__create_key,
            resolution_x,
            resolution_y,
            u_scalar,
            v_scalar,
        )

    def __init__(self,
                 class_private_create_key,
                 resolution_x: int,
                 resolution_y: int,
                 u_scalar: float,
                 v_scalar: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is CameraIntrinsics.__create_key, "Use create() method"

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        self.__vec_c = np.array([1.0,      0.0,      0.0], dtype=np.float32)
        self.__vec_u = np.array([0.0, u_scalar,      0.0], dtype=np.float32)
        self.__vec_v = np.array([0.0,      0.0, v_scalar], dtype=np.float32)

    @staticmethod
    def __pixel_vector_from_image_space(pixel: int,
                                        resolution: int,
                                        vec_base: np.ndarray) -> "tuple[bool, np.ndarray | None]":
        """
        Get u or v vector from pixel coordinate.
        """
        if pixel < 0:
            return False, None

        if resolution < 0:
            return False, None

        if pixel > resolution:
            return False, None

        if not is_vector_r3(vec_base):
            return False, None

        # Scaling factor with translation: 2 * p / r - 1 == (2 * p - r) / r
        # Codomain is from -1 to 1 inclusive
        scaling = float(2 * pixel - resolution) / float(resolution)

        vec_pixel = scaling * vec_base

        return True, vec_pixel

    def camera_space_from_image_space(self,
                                      pixel_x: int,
                                      pixel_y: int) -> "tuple[bool, np.ndarray | None]":
        """
        Pixel in image space to vector in camera space.
        """
        if pixel_x < 0:
            return False, None

        if pixel_y < 0:
            return False, None

        result, vec_pixel_u = self.__pixel_vector_from_image_space(
            pixel_x,
            self.resolution_x,
            self.__vec_u,
        )
        if not result:
            return False, None

        result, vec_pixel_v = self.__pixel_vector_from_image_space(
            pixel_y,
            self.resolution_y,
            self.__vec_v,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert vec_pixel_u is not None
        assert vec_pixel_v is not None

        vec_camera = self.__vec_c + vec_pixel_u + vec_pixel_v

        return True, vec_camera


class CameraDroneExtrinsics:
    """
    Camera in relation to drone.
    """
    __create_key = object()

    @classmethod
    def create(cls,
               camera_position_xyz: "tuple[float, float, float]",
               camera_orientation_ypr: "tuple[float, float, float]") \
            -> "tuple[bool, CameraDroneExtrinsics | None]":
        """
        camera_position_xyz: Camera position is x, y, z.
        camera_orientation_ypr: Camera orientation is yaw, pitch, roll.

        Both are relative to drone in NED system (x forward, y right, z down).
        Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order.
        """
        # Unpack parameters
        camera_x, camera_y, camera_z = camera_position_xyz
        camera_yaw, camera_pitch, camera_roll = camera_orientation_ypr

        vec_camera_on_drone_position = np.array([camera_x, camera_y, camera_z], dtype=np.float32)

        result, camera_to_drone_rotation_matrix = \
            create_rotation_matrix_from_orientation(
                camera_yaw,
                camera_pitch,
                camera_roll,
            )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert camera_to_drone_rotation_matrix is not None

        if not is_matrix_r3x3(camera_to_drone_rotation_matrix):
            return False, None

        return True, CameraDroneExtrinsics(
            cls.__create_key,
            vec_camera_on_drone_position,
            camera_to_drone_rotation_matrix,
        )

    def __init__(self,
                 class_private_create_key,
                 vec_camera_on_drone_position: np.ndarray,
                 camera_to_drone_rotation_matrix: np.ndarray):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is CameraDroneExtrinsics.__create_key, "Use create() method"

        self.vec_camera_on_drone_position = vec_camera_on_drone_position
        self.camera_to_drone_rotation_matrix = camera_to_drone_rotation_matrix
