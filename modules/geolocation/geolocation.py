"""
Converts image space into world space.
"""
import math

import cv2
import numpy as np

from .. import detection_in_world
from .. import merged_odometry_detections


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

        if fov_x <= 0.0:
            return False, None

        if fov_y <= 0.0:
            return False, None

        u_scalar = np.tan(fov_x / 2)
        if not np.isfinite(u_scalar):
            return False, None

        v_scalar = np.tan(fov_y / 2)
        if not np.isfinite(v_scalar):
            return False, None

        return True, CameraIntrinsics(cls.__create_key, resolution_x, resolution_y, fov_x, fov_y)

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

        if not Geolocation.is_vector_r3(vec_base):
            return False, None

        # Scaling factor with translation: 2 * p / r - 1 == (2 * p - r) / r
        # Codomain is [-1, 1]
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
               camera_x: float,
               camera_y: float,
               camera_z: float,
               camera_yaw: float,
               camera_pitch: float,
               camera_roll: float) -> "tuple[bool, CameraDroneExtrinsics | None]":
        """
        Camera position is in NED system (x forward, y right, z down).
        Camera rotation is in NED system (x forward, y right, z down).
        Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order.
        """
        vec_camera_position = np.array([camera_x, camera_y, camera_z], dtype=np.float32)

        result, camera_to_drone_rotation_matrix = Geolocation.create_rotation_matrix_from_orientation(
            camera_yaw,
            camera_pitch,
            camera_roll,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert camera_to_drone_rotation_matrix is not None

        if not Geolocation.is_matrix_r3x3(camera_to_drone_rotation_matrix):
            return False, None

        return True, CameraDroneExtrinsics(cls.__create_key, vec_camera_position, camera_to_drone_rotation_matrix)

    def __init__(self,
                 class_private_create_key,
                 vec_camera_position: np.ndarray,
                 camera_to_drone_rotation_matrix: np.ndarray):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is CameraDroneExtrinsics.__create_key, "Use create() method"

        self.vec_camera_position = vec_camera_position
        self.camera_to_drone_rotation_matrix = camera_to_drone_rotation_matrix


class Geolocation:
    """
    Converts image space into world space.
    """
    __create_key = object()

    __MIN_DOWN_COS_ANGLE = 0.1  # ~84°

    @classmethod
    def create(cls,
               camera_intrinsics: CameraIntrinsics,
               camera_drone_extrinsics: CameraDroneExtrinsics):
        """
        camera_intrinsics: Camera information without any outside space.
        camera_drone_extrinsics: Camera information related to the drone without any world space.
        """
        # Centre of each edge
        perspective_transform_sources = [
            [camera_intrinsics.resolution_x / 2, 0],
            [camera_intrinsics.resolution_x / 2, camera_intrinsics.resolution_y],
            [0, camera_intrinsics.resolution_y / 2],
            [camera_intrinsics.resolution_x, camera_intrinsics.resolution_y / 2],
        ]

        # Orientation in world space
        rotated_source_vectors = []
        for source in perspective_transform_sources:
            # Image space to camera space
            result, value = camera_intrinsics.camera_space_from_image_space(source[0], source[1])
            if not result:
                return False, None

            # Get Pylance to stop complaining
            assert value is not None

            # Camera space to world space (orientation only)
            vec_rotated_source = camera_drone_extrinsics @ value
            rotated_source_vectors.append(vec_rotated_source)

        return True, Geolocation(
            cls.__create_key,
            camera_intrinsics,
            camera_drone_extrinsics,
            perspective_transform_sources,
            rotated_source_vectors,
        )

    def __init__(self,
                 class_private_create_key,
                 camera_intrinsics: CameraIntrinsics,
                 camera_drone_extrinsics: CameraDroneExtrinsics,
                 perspective_transform_sources: "list[list[float]]",
                 rotated_source_vectors: "list[np.ndarray]"):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is Geolocation.__create_key, "Use create() method"

        self.__camera_intrinsics = camera_intrinsics
        self.__camera_drone_extrinsics = camera_drone_extrinsics
        self.__perspective_transform_sources = perspective_transform_sources
        self.__rotated_source_vectors = rotated_source_vectors

    @staticmethod
    def is_vector_r3(vec: np.ndarray) -> bool:
        """
        Checks if the numpy array is a vector in R^3 .
        """
        return vec.shape == (3)

    @staticmethod
    def is_matrix_r3x3(matrix: np.ndarray) -> bool:
        """
        Checks if the numpy array is a matrix in R^3x3
        """
        return matrix.shape == (3, 3)

    @staticmethod
    def create_rotation_matrix_from_orientation(yaw: float,
                                                pitch: float,
                                                roll: float) -> "tuple[bool, np.ndarray | None]":
        """
        Creates a rotation matrix from yaw pitch roll in NED system (x forward, y right, z down).
        Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order.
        See: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations

        yaw, pitch, roll are in radians from -pi to pi .
        """
        if yaw < -math.pi or yaw > math.pi:
            return False, None

        if pitch < -math.pi or pitch > math.pi:
            return False, None

        if roll < -math.pi or roll > math.pi:
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

    @staticmethod
    def ground_intersection_from_vector(vec_drone_position: np.ndarray,
                                        vec_camera_on_drone_position: np.ndarray,
                                        vec_down: np.ndarray) -> "tuple[bool, np.ndarray | None]":
        """
        Get 2D coordinates of where the downwards pointing vector intersects the ground.
        """
        if not Geolocation.is_vector_r3(vec_drone_position):
            return False, None

        if not Geolocation.is_vector_r3(vec_camera_on_drone_position):
            return False, None

        if not Geolocation.is_vector_r3(vec_down):
            return False, None

        vec_camera_position = vec_drone_position + vec_camera_on_drone_position
        if vec_camera_position[2] < 0.0:
            return False, None

        # Ensure vector is pointing down by checking angle
        # cos(angle) = a dot b / (||a|| * ||b||)
        vec_negative_z = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        cos_angle = np.dot(vec_down, vec_negative_z) / np.linalg.norm(vec_down)
        if cos_angle < Geolocation.__MIN_DOWN_COS_ANGLE:
            return False, None

        # Find scalar multiple for the vector to touch the ground (z/3rd component is 0)
        # Solve for s: o3 + s * d3 = 0
        scaling = -vec_camera_position[2] / vec_down[2]
        if scaling < 0.0:
            return False, None

        vec_ground = vec_camera_position + scaling * vec_down

        return True, vec_ground[:2]

    def run(self,
            detections: merged_odometry_detections.MergedOdometryDetections) \
        -> "tuple[bool, list[detection_in_world.DetectionInWorld] | None]":
        """
        Returns detections in world space.
        """
        # Generate projective perspective matrix
        drone_position = detections.drone_position

        raise NotImplementedError
