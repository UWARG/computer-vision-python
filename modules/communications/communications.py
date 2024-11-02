"""
Logs data and forwards it.
"""

import time

from .. import detection_in_world
from ..common.logger.modules import logger
from ..common.mavlink.modules import drone_odometry
from ..common.mavlink.modules import drone_odometry_local
from ..common.mavlink.modules import local_global_conversion


class Communications:
    """
    Currently logs data only.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        home_location: drone_odometry.DronePosition,
        local_logger: logger.Logger,
    ) -> "tuple[bool, Communications | None]":
        """
        Logs data and forwards it.

        home_location: Take-off location of drone.

        Returns: Success, class object.
        """

        return True, Communications(cls.__create_key, home_location, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        home_location: drone_odometry.DronePosition,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Communications.__create_key, "Use create() method"

        self.__home_location = home_location
        self.__logger = local_logger

    def run(
        self, detections_in_world: list[detection_in_world.DetectionInWorld]
    ) -> tuple[bool, list[detection_in_world.DetectionInWorld] | None]:

        detections_in_world_global = []
        for detection_in_world in detections_in_world:
            # TODO: Change this when the conversion interface is changed
            north = detection_in_world.centre[0]
            east = detection_in_world.centre[1]
            down = 0

            result, drone_position_local = drone_odometry_local.DronePositionLocal.create(
                north,
                east,
                down,
            )
            if not result:
                self.__logger.warning(
                    f"Could not convert DetectionInWorld to DronePositionLocal:\ndetection in world: {detection_in_world}"
                )
                return False, None

            result, detection_in_world_global = (
                local_global_conversion.drone_position_global_from_local(
                    self.__home_location, drone_position_local
                )
            )

            if not result:
                # Log nothing if at least one of the conversions failed
                self.__logger.warning(
                    f"drone_position_global_from_local conversion failed:\nhome_location: {self.__home_location}\ndrone_position_local: {drone_position_local}"
                )
                return False, None

            detections_in_world_global.append(detection_in_world_global)

        timestamp = time.time()
        self.__logger.info(f"{timestamp}: {detections_in_world_global}")

        return True, detections_in_world
