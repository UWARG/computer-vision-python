import time
from modules.common.logger.modules import logger
from modules.common.mavlink.modules.drone_odometry import DronePosition
from modules.detection_in_world import DetectionInWorld
from modules.flight_interface.local_global_conversion import detection_in_world_global_from_local


class Communications:
    """
    """
    __create_key = object()

    @classmethod
    def create(
        cls,
        local_logger: logger.Logger,
    ) -> "tuple[bool, Communications | None]":
        """
        Logs data and forwards it.
        """

        return True, Communications(cls.__create_key, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Communications.__create_key, "Use create() method"

        self.__logger = local_logger

    def run(
        self, detections_in_world: list[DetectionInWorld], home_location: DronePosition
    ) -> tuple[bool, list[DetectionInWorld] | None]:
        for detection_in_world in detections_in_world:
            result, detection_in_world_global = detection_in_world_global_from_local(
                home_location, detection_in_world
            )

            if not result:
                self.__logger.error("conversion failed")
                return False, detections_in_world

            self.__logger.info(str(time.time()) + ": " + str(detection_in_world_global))

        return True, detections_in_world
