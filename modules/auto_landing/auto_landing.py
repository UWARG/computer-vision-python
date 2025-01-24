""" 
Auto-landing script.
"""

import math

from ..common.modules.logger import logger
from .. import detections_and_time


class AutoLanding:
    """
    Auto-landing script.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        local_logger: logger.Logger,
    ) -> "tuple [bool, AutoLanding | None ]":
        """
        fov_x: The horizontal camera field of view in degrees.
        fov_y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.
        """
        local_logger.info("", True)
        return True, AutoLanding(cls.__create_key, fov_x, fov_y, im_w, im_h, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        height_agl: float,
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
        self.height_agl = height_agl
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

        self.__logger.info(f"X angle (rad): {angle_x}", True)
        self.__logger.info(f"Y angle (rad): {angle_y}", True)

        x_dist = math.tan(angle_x) * self.height_agl
        y_dist = math.tan(angle_y) * self.height_agl
        ground_hyp = (x_dist**2 + y_dist**2) ** 0.5
        self.__logger.info(f"Required horizontal correction (m): {ground_hyp}", True)
        target_to_vehicle_dist = (ground_hyp**2 + self.height_agl**2) ** 0.5
        self.__logger.info(f"Distance from vehicle to target (m): {target_to_vehicle_dist}", True)

        return True, (angle_x, angle_y, target_to_vehicle_dist)
