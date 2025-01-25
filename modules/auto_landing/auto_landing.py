""" 
Auto-landing script that calculates the necessary parameters 
for use with LANDING_TARGET MAVLink command. 
"""

import math
import time

from .. import detections_and_time
from ..common.modules.logger import logger

# from .. import merged_odometry_detections


class AutoLanding:
    """
    Auto-landing script that calculates the necessary parameters
    for use with LANDING_TARGET MAVLink command.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        period: float,
        local_logger: logger.Logger,
    ) -> "tuple [bool, AutoLanding | None ]":
        """
        fov_x: The horizontal camera field of view in degrees.
        fov_y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.

        Returns an AutoLanding object.
        """

        return True, AutoLanding(cls.__create_key, fov_x, fov_y, im_h, im_w, period, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        period: float,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is AutoLanding.__create_key, "Use create() method"

        self.fov_x = fov_x
        self.fov_y = fov_y
        self.im_h = im_h
        self.im_w = im_w
        self.period = period
        self.__logger = local_logger

    def run(
        self, bounding_box: detections_and_time.Detection
    ) -> "tuple[bool, tuple[float, float, float]]":
        """
        Calculates the angles in radians of the bounding box based on its center.

        bounding_box: A detections and time object.

        Return: Tuple of the x and y angles in radians respectively and the target distance in meters.
        """

        x_center, y_center = bounding_box.get_centre()

        angle_x = (x_center - self.im_w / 2) * (self.fov_x * (math.pi / 180)) / self.im_w
        angle_y = (y_center - self.im_h / 2) * (self.fov_y * (math.pi / 180)) / self.im_h

        height_agl = 0

        x_dist = math.tan(angle_x) * height_agl
        y_dist = math.tan(angle_y) * height_agl
        ground_hyp = (x_dist**2 + y_dist**2) ** 0.5
        target_to_vehicle_dist = (ground_hyp**2 + height_agl**2) ** 0.5

        self.__logger.info(
            f"X angle: {angle_x} Y angle: {angle_y}\nRequired horizontal correction: {ground_hyp} Distance from vehicle to target: {target_to_vehicle_dist}",
            True,
        )

        time.sleep(self.period)

        return True, (angle_x, angle_y, target_to_vehicle_dist)
