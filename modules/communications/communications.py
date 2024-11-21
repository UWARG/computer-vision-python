"""
Logs data and forwards it.
"""

import time

from .. import object_in_world
from ..common.modules.logger import logger
from ..common.modules import position_global
from ..common.modules import position_local
from ..common.modules.mavlink import local_global_conversion


class Communications:
    """
    Currently logs data only.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        home_position: position_global.PositionGlobal,
        local_logger: logger.Logger,
    ) -> "tuple[bool, Communications | None]":
        """
        Logs data and forwards it.

        home_position: Take-off position of drone.

        Returns: Success, class object.
        """

        return True, Communications(cls.__create_key, home_position, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        home_position: position_global.PositionGlobal,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Communications.__create_key, "Use create() method"

        self.__home_position = home_position
        self.__logger = local_logger

    def run(
        self, objects_in_world: list[object_in_world.ObjectInWorld],
    ) -> tuple[bool, list[object_in_world.ObjectInWorld] | None]:

        objects_in_world_global = []
        for object_in_world in objects_in_world:
            # TODO: Change this when the conversion interface is changed
            north = object_in_world.location_x
            east = object_in_world.location_y
            down = 0

            result, object_position_local = position_local.PositionLocal.create(
                north,
                east,
                down,
            )
            if not result:
                self.__logger.warning(
                    f"Could not convert ObjectInWorld to PositionLocal:\object in world: {object_in_world}"
                )
                return False, None

            result, object_in_world_global = (
                local_global_conversion.drone_position_global_from_local(
                    self.__home_position, object_position_local
                )
            )

            if not result:
                # Log nothing if at least one of the conversions failed
                self.__logger.warning(
                    f"drone_position_global_from_local conversion failed:\nhome_position: {self.__home_position}\ndrone_position_local: {object_position_local}"
                )
                return False, None

            objects_in_world_global.append(object_in_world_global)

        timestamp = time.time()
        self.__logger.info(f"{timestamp}: {objects_in_world_global}")

        return True, objects_in_world
