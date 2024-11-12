"""
Converts image space into world space in a more rudimentary sense
"""

import numpy as np

from .. import detection_in_world
from .. import merged_odometry_detections
from ..common.modules.logger import logger
from .geolocation import Geolocation

class RudimentaryGeolocation(Geolocation):
    """
    Converts image space into world space.
    """

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

        # Since camera points down, the rotation matrix will be this
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
