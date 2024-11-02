import time
from ..common.mavlink.modules import drone_odometry
from modules.common.logger.modules import logger
from modules.common.mavlink.modules.drone_odometry import DronePosition
from modules.detection_in_world import DetectionInWorld
from modules.flight_interface.local_global_conversion import detection_in_world_global_from_local


class Communications:
    """ """

    __create_key = object()

    @classmethod
    def create(
        cls,
        home_location: drone_odometry.DronePosition,
        local_logger: logger.Logger,
    ) -> "tuple[bool, Communications | None]":
        """
        Logs data and forwards it.
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
        self, detections_in_world: list[DetectionInWorld]
    ) -> tuple[bool, list[DetectionInWorld] | None]:

        detections_in_world_global = []
        for detection_in_world in detections_in_world:
            result, detection_in_world_global = detection_in_world_global_from_local(
                self.__home_location, detection_in_world
            )

            if not result:
                # Log nothing if at least one of the conversions failed
                self.__logger.error("conversion failed")
                return False, detections_in_world

            detections_in_world_global.append(detection_in_world_global)

        timestamp = time.time()
        self.__logger.info(f"{timestamp}: {detections_in_world_global}")

        return True, detections_in_world
