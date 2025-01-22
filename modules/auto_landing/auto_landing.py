import math

from ..common.modules.logger import logger
from .. import detections_and_time

class AutoLanding:
    """"
    Auto-landing script. 
    """
    __create_key = object ()

    @classmethod
    def create (
        cls,
        FOV_X: float,
        FOV_Y: float,
        local_logger: logger.Logger,
    ) -> "tuple [bool, AutoLanding | None ]":
        """"
        FOV_X: The horizontal camera field of view in degrees.
        FOV_Y: The vertical camera field of view in degrees.

        """
        
        return True, AutoLanding(
            cls.__create_key, FOV_X, FOV_Y, local_logger
        )
    

    def __init__ (
            self,
            class_private_create_key: object,
            FOV_X: float,
            FOV_Y: float,
            local_logger: logger.Logger
        ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is AutoLanding.__create_key, "Use create() method"

        self.FOV_X = FOV_X
        self.FOV_Y = FOV_Y
        self.__logger = local_logger

    def run (self, 
             height: float
        ) -> "tuple[float, float, float]":
        """
        Calculates the angles in radians of the bounding box based on its center.

        height: Height above ground level in meters.

        Return: Tuple of the x and y angles in radians respectively and the target distance in meters.
        """
        x_center, y_center = detections_and_time.Detection.get_centre()

        top_left, top_right, _, bottom_right = detections_and_time.Detection.get_corners()

        im_w = abs(top_right - top_left)
        im_h = abs(top_right - bottom_right)

        angle_x = (x_center - im_w / 2) * (self.FOV_X * (math.pi / 180)) / im_w
        angle_y = (y_center - im_h / 2) * (self.FOV_Y * (math.pi / 180)) / im_h

        self.__logger.info(f"X angle (rad): {angle_x}", True)
        self.__logger.info(f"Y angle (rad): {angle_y}", True)

        x_dist = math.tan(angle_x) * height
        y_dist = math.tan(angle_y) * height
        ground_hyp = (x_dist**2 + y_dist**2)**0.5
        self.__logger.info(f"Required horizontal correction (m): {ground_hyp}", True)
        target_to_vehicle_dist = (ground_hyp**2 + height**2)**0.5
        self.__logger.info(f"Distance from vehicle to target (m): {target_to_vehicle_dist}", True)

        return angle_x, angle_y, target_to_vehicle_dist


    