""" 
Auto-landing script that calculates the necessary parameters 
for use with LANDING_TARGET MAVLink command. 
"""

import math

from .. import merged_odometry_detections
from ..common.modules.logger import logger


class AutoLandingInformation:
    """
    Information necessary for the LANDING_TARGET MAVLink command.
    """

    def __init__(self, angle_x: float, angle_y: float, target_dist: float) -> None:
        """
        Information necessary for the LANDING_TARGET MAVLink command.

        angle_x: Angle of the x coordinate of the bounding box within -π to π (rads).
        angle_y: Angle of the y coordinate of the bounding box within -π to π (rads).
        target_dist: Horizontal distance of vehicle to target (meters).
        """

        self.angle_x = angle_x
        self.angle_y = angle_y
        self.target_dist = target_dist


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
        local_logger: logger.Logger,
    ) -> "tuple [bool, AutoLanding | None ]":
        """
        fov_x: The horizontal camera field of view in degrees.
        fov_y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.

        Returns an AutoLanding object.
        """

        return True, AutoLanding(cls.__create_key, fov_x, fov_y, im_h, im_w, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
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
        self.__logger = local_logger

    def run(
        self, odometry_detections: merged_odometry_detections.MergedOdometryDetections
    ) -> "tuple[bool, AutoLandingInformation]":
        """
        Calculates the x and y angles in radians of the bounding box based on its center.

        odometry_detections: A merged odometry dectections object.

        Returns an AutoLandingInformation object.
        """

        # TODO: Devise better algorithm to pick which detection to land at if several are detected
        x_center, y_center = odometry_detections.detections[0].get_centre()

        angle_x = (x_center - self.im_w / 2) * (self.fov_x * (math.pi / 180)) / self.im_w
        angle_y = (y_center - self.im_h / 2) * (self.fov_y * (math.pi / 180)) / self.im_h

        # This is height above ground level (AGL)
        # down is how many meters down you are from home position, which is on the ground
        height_agl = odometry_detections.odometry_local.position.down * -1

        x_dist = math.tan(angle_x) * height_agl
        y_dist = math.tan(angle_y) * height_agl
        ground_hyp = (x_dist**2 + y_dist**2) ** 0.5
        target_to_vehicle_dist = (ground_hyp**2 + height_agl**2) ** 0.5

        self.__logger.info(
            f"X angle: {angle_x} Y angle: {angle_y}\nRequired horizontal correction: {ground_hyp} Distance from vehicle to target: {target_to_vehicle_dist}",
            True,
        )

        return True, AutoLandingInformation(angle_x, angle_y, target_to_vehicle_dist)
