"""
Converts image space into world space.
"""

import cv2
import numpy as np

from . import camera_properties
from .. import detection_in_world
from .. import detections_and_time
from .. import merged_odometry_detections
from ..common.modules.logger import logger


class RudimentaryGeolocation:
    """
    Converts image space into world space.
    """

    __create_key = object()

    __MIN_DOWN_COS_ANGLE = 0.1  # radians, ~84°

    @classmethod
    def create(
        cls,
        camera_intrinsics: camera_properties.CameraIntrinsics,
        camera_drone_extrinsics: camera_properties.CameraDroneExtrinsics,
        local_logger: logger.Logger,
    ) -> "tuple[bool, RudimentaryGeolocation | None]":
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
                local_logger.error(
                    f"Rotated source vector could not be created for source: {source}"
                )
                return False, None

            # Get Pylance to stop complaining
            assert value is not None

            # Camera space to world space (orientation only)
            vec_rotated_source = camera_drone_extrinsics.camera_to_drone_rotation_matrix @ value
            rotated_source_vectors.append(vec_rotated_source)

        return True, RudimentaryGeolocation(
            cls.__create_key,
            camera_drone_extrinsics,
            perspective_transform_sources,
            rotated_source_vectors,
            local_logger,
        )

    def __init__(
        self,
        class_private_create_key: object,
        camera_drone_extrinsics: camera_properties.CameraDroneExtrinsics,
        perspective_transform_sources: "list[list[float]]",
        rotated_source_vectors: "list[np.ndarray]",
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is RudimentaryGeolocation.__create_key, "Use create() method"

        self.__camera_drone_extrinsics = camera_drone_extrinsics
        self.__perspective_transform_sources = perspective_transform_sources
        self.__rotated_source_vectors = rotated_source_vectors
        self.__logger = local_logger

    # TODO: We need to convert both these fucntions to work for our case
    @staticmethod
    def __ground_intersection_from_vector(
        vec_camera_in_world_position: np.ndarray,
        vec_down: np.ndarray,
        local_logger: logger.Logger,
    ) -> "tuple[bool, np.ndarray | None]":
        """
        Get 2D coordinates of where the downwards pointing vector intersects the ground.
        """
        if not camera_properties.is_vector_r3(vec_camera_in_world_position):
            local_logger.error("Camera position in world space is not a vector in R3")
            return False, None

        if not camera_properties.is_vector_r3(vec_down):
            local_logger.error("Rotated source vector in world space is not a vector in R3")
            return False, None

        # Check camera above ground
        if vec_camera_in_world_position[2] > 0.0:
            local_logger.error("Camera is underground")
            return False, None

        # Ensure vector is pointing down by checking angle
        # cos(angle) = a dot b / (||a|| * ||b||)
        vec_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cos_angle = np.dot(vec_down, vec_z) / np.linalg.norm(vec_down)
        if cos_angle < RudimentaryGeolocation.__MIN_DOWN_COS_ANGLE:
            local_logger.error(
                f"Rotated source vector in world space is not pointing down, cos(angle) = {cos_angle}"
            )
            return False, None

        # Find scalar multiple for the vector to touch the ground (z/3rd component is 0)
        # Solve for s: o3 + s * d3 = 0
        scaling = -vec_camera_in_world_position[2] / vec_down[2]
        if scaling < 0.0:
            local_logger.error(f"Scaling value is negative, scaling = {scaling}")
            return False, None

        vec_ground = vec_camera_in_world_position + scaling * vec_down

        return True, vec_ground[:2]

    def __get_perspective_transform_matrix(
        self, drone_rotation_matrix: np.ndarray, drone_position_ned: np.ndarray
    ) -> "tuple[bool, np.ndarray | None]":
        """
        Calculates the destination points, then uses OpenCV to get the matrix.
        """
        if not camera_properties.is_matrix_r3x3(drone_rotation_matrix):
            self.__logger.error("Drone rotation matrix is not a 3 x 3 matrix")
            return False, None

        if not camera_properties.is_vector_r3(drone_position_ned):
            self.__logger.error("Drone position in local space is not a vector in R3")
            return False, None

        # Get the vectors in world space
        vec_downs = []
        for vector in self.__rotated_source_vectors:
            vec_down = drone_rotation_matrix @ vector
            vec_downs.append(vec_down)

        # Get the camera position in world space
        vec_camera_position = (
            drone_position_ned
            + drone_rotation_matrix @ self.__camera_drone_extrinsics.vec_camera_on_drone_position
        )

        # Find the points on the ground
        ground_points = []
        for vec_down in vec_downs:
            result, ground_point = self.__ground_intersection_from_vector(
                vec_camera_position,
                vec_down,
                self.__logger,
            )
            if not result:
                return False, None

            ground_points.append(ground_point)

        # Get the image to ground mapping
        src = np.array(self.__perspective_transform_sources, dtype=np.float32)
        dst = np.array(ground_points, dtype=np.float32)
        try:
            matrix = cv2.getPerspectiveTransform(  # type: ignore
                src,
                dst,
            )
        # All exceptions must be caught and logged as early as possible
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            self.__logger.error(f"Could not get perspective transform matrix: {e}")
            return False, None

        return True, matrix

    @staticmethod
    def __convert_detection_to_world_from_image(
        detection: detections_and_time.Detection,
        perspective_transform_matrix: np.ndarray,
        local_logger: logger.Logger,
    ) -> "tuple[bool, detection_in_world.DetectionInWorld | None]":
        """
        Applies the transform matrix to the detection.
        perspective_transform_matrix: Element in last row and column must be 1 .
        """
        if not camera_properties.is_matrix_r3x3(perspective_transform_matrix):
            local_logger.error("Perspective transform matrix is not a 3 x 3 matrix")
            return False, None

        if not np.allclose(perspective_transform_matrix[2][2], 1.0):
            local_logger.error(
                "Perspective transform matrix bottom right element is not close to 1.0"
            )
            return False, None

        centre = detection.get_centre()
        top_left, top_right, bottom_left, bottom_right = detection.get_corners()

        input_centre = np.array([centre[0], centre[1], 1.0], dtype=np.float32)
        # More efficient to multiply a matrix than looping over the points
        # Transpose to columns from rows
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

        # Normalize by last element
        # Homogeneous/perspective divide:
        # https://en.wikipedia.org/wiki/Transformation_matrix#Perspective_projection
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

        # Divide each row by vector element:
        # https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-96.php
        output_normalized = output_vertices / vec_last_element[:, None]
        if not np.isfinite(output_normalized).all():
            local_logger.error("Normalized output is infinite")
            return False, None

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

    def run(
        self, detections: merged_odometry_detections.MergedOdometryDetections
    ) -> "tuple[bool, list[detection_in_world.DetectionInWorld] | None]":
        """
        Runs detections in world space
        """
        if detections.odometry_local.position.down >= 0.0:
            self.__logger.error("Drone is underground")
            return False, None

        drone_position_ned = np.array(
            [
                detections.odometry_local.position.north,
                detections.odometry_local.position.east,
                detections.odometry_local.position.down,
            ],
            dtype=np.float32,
        )

        # Since camera points down, the rotation matrix will be
        # Calculated assuming pitch=-pi/2 and yaw=roll=0
        drone_rotation_matrix = np.array(
            [
                [0.0, 0.0,-1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        result, perspective_transform_matrix = self.__get_perspective_transform_matrix(
            drone_rotation_matrix,
            drone_position_ned,
        )
        if not result:
            return False, None

        detections_in_world = []
        for detection in detections.detections:
            result, detection_world = self.__convert_detection_to_world_from_image(
                detection,
                perspective_transform_matrix,
                self.__logger,
            )
            # Partial data not allowed
            if not result:
                return False, None
            detections_in_world.append(detection_world)
            self.__logger.info(detection_world)

        return True, detections_in_world
