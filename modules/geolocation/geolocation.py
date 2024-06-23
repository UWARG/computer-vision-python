"""
Converts image space into world space.
"""

import cv2
import numpy as np

from . import camera_properties
from .. import detection_in_world
from .. import detections_and_time
from .. import merged_odometry_detections


class Geolocation:
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
    ) -> "tuple[bool, Geolocation | None]":
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
                print("1")
                return False, None

            # Get Pylance to stop complaining
            assert value is not None

            # Camera space to world space (orientation only)
            vec_rotated_source = camera_drone_extrinsics.camera_to_drone_rotation_matrix @ value
            rotated_source_vectors.append(vec_rotated_source)

        return True, Geolocation(
            cls.__create_key,
            camera_drone_extrinsics,
            perspective_transform_sources,
            rotated_source_vectors,
        )

    def __init__(
        self,
        class_private_create_key: object,
        camera_drone_extrinsics: camera_properties.CameraDroneExtrinsics,
        perspective_transform_sources: "list[list[float]]",
        rotated_source_vectors: "list[np.ndarray]",
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Geolocation.__create_key, "Use create() method"

        self.__camera_drone_extrinsics = camera_drone_extrinsics
        self.__perspective_transform_sources = perspective_transform_sources
        self.__rotated_source_vectors = rotated_source_vectors

    @staticmethod
    def __ground_intersection_from_vector(
        vec_camera_in_world_position: np.ndarray, vec_down: np.ndarray
    ) -> "tuple[bool, np.ndarray | None]":
        """
        Get 2D coordinates of where the downwards pointing vector intersects the ground.
        """
        if not camera_properties.is_vector_r3(vec_camera_in_world_position):
            print("2")
            return False, None

        if not camera_properties.is_vector_r3(vec_down):
            print("3")
            return False, None

        # Check camera above ground
        if vec_camera_in_world_position[2] > 0.0:
            print("4")
            return False, None

        # Ensure vector is pointing down by checking angle
        # cos(angle) = a dot b / (||a|| * ||b||)
        vec_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cos_angle = np.dot(vec_down, vec_z) / np.linalg.norm(vec_down)
        angle = np.arccos(cos_angle) * (180.0 / np.pi)
        print(f"cos angle = {cos_angle} | angle = {angle}")
        # if cos_angle < Geolocation.__MIN_DOWN_COS_ANGLE:
        #     print("5")
        #     print(f"cos angle if statement = {cos_angle}")  # cos angle = -0.7073870897293091
        #     return False, None
        # else:
        #     print(f"cos angle else statement = {cos_angle}") # cos angle = 0.7068257927894592

        # Find scalar multiple for the vector to touch the ground (z/3rd component is 0)
        # Solve for s: o3 + s * d3 = 0
        scaling = -vec_camera_in_world_position[2] / vec_down[2]
        # if scaling < 0.0:
        #     print("6")
        #     print(scaling)  
        #     return False, None

        print(f"scaling = {scaling}")  # scaling = 2.030820369720459

        vec_ground = vec_camera_in_world_position + scaling * vec_down

        return True, vec_ground[:2]

    def __get_perspective_transform_matrix(
        self, drone_rotation_matrix: np.ndarray, drone_position_ned: np.ndarray
    ) -> "tuple[bool, np.ndarray | None]":
        """
        Calculates the destination points, then uses OpenCV to get the matrix.
        """
        if not camera_properties.is_matrix_r3x3(drone_rotation_matrix):
            print("7")
            return False, None

        if not camera_properties.is_vector_r3(drone_position_ned):
            print("8")
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
            print(f"vec down = {vec_down}")
            result, ground_point = self.__ground_intersection_from_vector(
                vec_camera_position,
                vec_down,
            )
            if not result:
                print("9")
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
        # pylint: disable-next=bare-except
        except:
            # TODO: Logging
            print("10")
            return False, None

        return True, matrix

    @staticmethod
    def __convert_detection_to_world_from_image(
        detection: detections_and_time.Detection, perspective_transform_matrix: np.ndarray
    ) -> "tuple[bool, detection_in_world.DetectionInWorld | None]":
        """
        Applies the transform matrix to the detection.
        perspective_transform_matrix: Element in last row and column must be 1 .
        """
        if not camera_properties.is_matrix_r3x3(perspective_transform_matrix):
            print("11")
            return False, None

        if not np.allclose(perspective_transform_matrix[2][2], 1.0):
            print("12")
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
            print("13")
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
            print("14")
            return False, None

        return True, detection_world

    def run(
        self, detections: merged_odometry_detections.MergedOdometryDetections
    ) -> "tuple[bool, list[detection_in_world.DetectionInWorld] | None]":
        """
        Returns detections in world space.
        """
        # Camera position in world (NED system)
        # Cannot be underground
        if detections.odometry_local.position.down >= 0.0:
            print("15")
            print(detections.odometry_local.position)
            return False, None

        drone_position_ned = np.array(
            [
                detections.odometry_local.position.north,
                detections.odometry_local.position.east,
                detections.odometry_local.position.down,
            ],
            dtype=np.float32,
        )

        # Generate projective perspective matrix
        # Camera rotation in world
        result, drone_rotation_matrix = camera_properties.create_rotation_matrix_from_orientation(
            detections.odometry_local.orientation.orientation.yaw,
            detections.odometry_local.orientation.orientation.pitch,
            detections.odometry_local.orientation.orientation.roll,
        )
        if not result:
            print("16")
            return False, None

        # Get Pylance to stop complaining
        assert drone_rotation_matrix is not None

        result, perspective_transform_matrix = self.__get_perspective_transform_matrix(
            drone_rotation_matrix,
            drone_position_ned,
        )
        if not result:
            print("17")
            return False, None

        # Get Pylance to stop complaining
        assert perspective_transform_matrix is not None

        detections_in_world = []
        calculated_center_x = 0
        calculated_center_y = 0
        for detection in detections.detections:
            result, detection_world = self.__convert_detection_to_world_from_image(
                detection,
                perspective_transform_matrix,
            )
            # Partial data not allowed
            if not result:
                print("18")
                return False, None
            
            for vertice in detection_world.vertices:
                calculated_center_x = calculated_center_x + vertice[0]
                calculated_center_y = calculated_center_y + vertice[1]

            calculated_center_x = calculated_center_x / 4.0
            calculated_center_y = calculated_center_y / 4.0

            print(f"calculated centre = ({calculated_center_x}, {calculated_center_y})")

            detections_in_world.append(detection_world)

        return True, detections_in_world
