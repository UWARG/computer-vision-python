"""
Converts image space into world space.
"""
import math

import cv2
import numpy as np

from .. import detection_in_world
from .. import detections_and_time
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
            Geolocation.create_rotation_matrix_from_orientation(
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

        return True, CameraDroneExtrinsics(
            cls.__create_key, vec_camera_on_drone_position,
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
        # list[list[float]] required for OpenCV
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
            vec_rotated_source = camera_drone_extrinsics.camera_to_drone_rotation_matrix @ value
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
        return vec.shape == (3,)

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
    def __ground_intersection_from_vector(vec_camera_in_world_position: np.ndarray,
                                          vec_down: np.ndarray) \
            -> "tuple[bool, np.ndarray | None]":
        """
        Get 2D coordinates of where the downwards pointing vector intersects the ground.
        """
        if not Geolocation.is_vector_r3(vec_camera_in_world_position):
            return False, None

        if not Geolocation.is_vector_r3(vec_down):
            return False, None

        # Check camera above ground
        if vec_camera_in_world_position[2] > 0.0:
            return False, None

        # Ensure vector is pointing down by checking angle
        # cos(angle) = a dot b / (||a|| * ||b||)
        vec_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cos_angle = np.dot(vec_down, vec_z) / np.linalg.norm(vec_down)
        if cos_angle < Geolocation.__MIN_DOWN_COS_ANGLE:
            return False, None

        # Find scalar multiple for the vector to touch the ground (z/3rd component is 0)
        # Solve for s: o3 + s * d3 = 0
        scaling = -vec_camera_in_world_position[2] / vec_down[2]
        if scaling < 0.0:
            return False, None

        vec_ground = vec_camera_in_world_position + scaling * vec_down

        return True, vec_ground[:2]

    def __get_perspective_transform_matrix(self,
                                           drone_rotation_matrix: np.ndarray,
                                           drone_position_ned: np.ndarray) \
            -> "tuple[bool, np.ndarray | None]":
        """
        Calculates the destination points, then uses OpenCV to get the matrix.
        """
        if not self.is_matrix_r3x3(drone_rotation_matrix):
            return False, None

        if not self.is_vector_r3(drone_position_ned):
            return False, None

        # Get the vectors in world space
        camera_overall_rotation_matrix = \
            drone_rotation_matrix @ self.__camera_drone_extrinsics.camera_to_drone_rotation_matrix

        vec_downs = []
        for vector in self.__rotated_source_vectors:
            vec_down = camera_overall_rotation_matrix @ vector
            vec_downs.append(vec_down)

        # Get the camera position in world space
        vec_camera_position = \
              drone_position_ned \
            + drone_rotation_matrix @ self.__camera_drone_extrinsics.vec_camera_on_drone_position

        # Find the points on the ground
        ground_points = []
        for vec_down in vec_downs:
            result, ground_point = self.__ground_intersection_from_vector(
                vec_camera_position,
                vec_down,
            )
            if not result:
                return False, None

            ground_points.append(ground_point)

        # Get the image to ground mapping
        src = np.array(self.__perspective_transform_sources, dtype=np.float32)
        dst = np.array(ground_points, dtype=np.float32)
        try:
            # Pylint does not like the OpenCV module
            # pylint: disable=no-member
            matrix = cv2.getPerspectiveTransform(
                src,
                dst,
            )
            # pylint: enable=no-member
        # All exceptions must be caught and logged as early as possible
        # pylint: disable=bare-except
        except:
            # TODO: Logging
            return False, None
        # pylint: enable=bare-except

        return True, matrix

    @staticmethod
    def __convert_detection_to_world_from_image(detection: detections_and_time.Detection,
                                                perspective_transform_matrix: np.ndarray) \
            -> "tuple[bool, detection_in_world.DetectionInWorld | None]":
        """
        Applies the transform matrix to the detection.
        """
        if not Geolocation.is_matrix_r3x3(perspective_transform_matrix):
            return False, None

        centre = detection.get_centre()
        top_left, top_right, bottom_left, bottom_right = detection.get_corners()

        input_centre = np.array([centre[0], centre[1], 1.0], dtype=np.float32)
        # More efficient to multiply a matrix than looping over the points
        # Transpose to columns from rowss
        input_vertices = np.array(
            [
                [top_left[0], top_left[1], 1.0],
                [top_right[0], top_right[1], 1.0],
                [bottom_left[0], bottom_left[1], 1.0],
                [bottom_right[0], bottom_right[1], 1.0],
            ],
            dtype=np.float32,
        ).T

        # Single row/column does not need transpose
        output_centre = perspective_transform_matrix @ input_centre
        # Transpose back to rows from columns
        output_vertices = (perspective_transform_matrix @ input_vertices).T

        ground_centre = np.array(
            [
                output_centre[0] / output_centre[2],
                output_centre[1] / output_centre[2],
            ],
            dtype=np.float32,
        )

        # Normalize each row by its last element
        # Slice to get the last element of each row
        vec_last_element = output_vertices[:, 2]
        # Divide each row by vector element
        output_normalized = output_vertices / vec_last_element[:,None]
        # Slice to remove the last element of each row
        ground_vertices = output_normalized[:, :2]

        result, detection_world = detection_in_world.DetectionInWorld.create(
            ground_vertices,
            ground_centre,
            detection.label,
            detection.confidence,
        )
        if not result:
            return False, None

        return True, detection_world

    def run(self,
            detections: merged_odometry_detections.MergedOdometryDetections) \
        -> "tuple[bool, list[detection_in_world.DetectionInWorld] | None]":
        """
        Returns detections in world space.
        """
        # Generate projective perspective matrix
        # Camera rotation in world
        result, drone_rotation_matrix = self.create_rotation_matrix_from_orientation(
            detections.drone_orientation.yaw,
            detections.drone_orientation.pitch,
            detections.drone_orientation.roll,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert drone_rotation_matrix is not None

        # Camera position in world
        # Convert to NED system
        drone_position_ned = np.array(
            [
                detections.drone_position.position_x,
                detections.drone_position.position_y,
                -detections.drone_position.altitude,
            ],
            dtype=np.float32,
        )

        result, perspective_transform_matrix = self.__get_perspective_transform_matrix(
            drone_rotation_matrix,
            drone_position_ned,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert perspective_transform_matrix is not None

        detections_in_world = []
        for detection in detections.detections:
            result, detection_world = self.__convert_detection_to_world_from_image(
                detection,
                perspective_transform_matrix,
            )
            # Partial data not allowed
            if not result:
                return False, None
            detections_in_world.append(detection_world)

        return True, detections_in_world
