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
