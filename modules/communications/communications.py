"""
Logs data and forwards it.
"""

import time

from .. import object_in_world
from ..common.modules import position_global
from ..common.modules import position_local
from ..common.modules.logger import logger
from ..common.modules.data_encoding import message_encoding_decoding
from ..common.modules.data_encoding import worker_enum
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
        worker_id: int,
    ) -> "tuple[True, Communications] | tuple[False, None]":
        """
        Logs data and forwards it.

        home_position: Take-off position of drone.

        Returns: Success, class object.
        """

        return True, Communications(cls.__create_key, home_position, local_logger, worker_id)

    def __init__(
        self,
        class_private_create_key: object,
        home_position: position_global.PositionGlobal,
        local_logger: logger.Logger,
        worker_id: int,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Communications.__create_key, "Use create() method"

        self.__home_position = home_position
        self.__logger = local_logger
        self.__worker_id = worker_id

    def run(
        self,
        objects_in_world: list[object_in_world.ObjectInWorld],
    ) -> tuple[True, list[bytes]] | tuple[False, None]:

        objects_in_world_global = []
        for object_in_world in objects_in_world:
            # We assume detected objects are on the ground
            north = object_in_world.location_x
            east = object_in_world.location_y
            down = 0.0

            result, object_position_local = position_local.PositionLocal.create(
                north,
                east,
                down,
            )
            if not result:
                self.__logger.warning(
                    f"Could not convert ObjectInWorld to PositionLocal:\nobject in world: {object_in_world}"
                )
                return False, None

            result, object_in_world_global = (
                local_global_conversion.position_global_from_position_local(
                    self.__home_position, object_position_local
                )
            )
            if not result:
                # Log nothing if at least one of the conversions failed
                self.__logger.warning(
                    f"position_global_from_position_local conversion failed:\nhome_position: {self.__home_position}\nobject_position_local: {object_position_local}"
                )
                return False, None

            objects_in_world_global.append(object_in_world_global)

        self.__logger.info(f"{time.time()}: {objects_in_world_global}")

        encoded_position_global_objects = []
        for object in object_in_world_global:

            result, message = message_encoding_decoding.encode_position_global(
                self.__worker_id, object
            )
            if not result:
                self.__logger.warning("Conversion from PositionGlobal to bytes failed", True)
                return False, None

            encoded_position_global_objects.append(message)

        return True, encoded_position_global_objects
