"""
Creates flight controller and combines odometry data and timestamp.
"""

from .. import decision_command
from .. import odometry_and_time
from ..common.modules.logger import logger
from ..common.modules import position_global
from ..common.modules import position_local
from ..common.modules.mavlink import flight_controller
from ..common.modules.mavlink import local_global_conversion


class FlightInterface:
    """
    Create flight controller and combines odometry data and timestamp.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        address: str,
        timeout_home: float,
        baud_rate: int,
        local_logger: logger.Logger,
    ) -> "tuple[bool, FlightInterface | None]":
        """
        address: TCP address or port.
        timeout_home: Timeout for home location in seconds.
        baud_rate: Baud rate for the connection.
        """
        result, controller = flight_controller.FlightController.create(address, baud_rate)
        if not result:
            local_logger.error("controller could not be created", True)
            return False, None

        # Get Pylance to stop complaining
        assert controller is not None

        result, home_position = controller.get_home_position(timeout_home)
        if not result:
            local_logger.error("home_position could not be created", True)
            return False, None

        # Get Pylance to stop complaining
        assert home_position is not None

        local_logger.info(f"Home position: {home_position}", True)

        return True, FlightInterface(cls.__create_key, controller, home_position, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        controller: flight_controller.FlightController,
        home_position: position_global.PositionGlobal,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller
        self.__home_position = home_position
        self.__logger = local_logger

    def get_home_position(self) -> position_global.PositionGlobal:
        """
        Accessor for home position.
        """
        return self.__home_position

    def run(self, message: bytes) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = self.controller.get_odometry()
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        result, odometry_local = local_global_conversion.drone_odometry_local_from_global(
            self.__home_position,
            odometry,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_local is not None

        result, odometry_and_time_object = odometry_and_time.OdometryAndTime.create(odometry_local)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_and_time_object is not None

        self.__logger.info(str(odometry_and_time_object), True)

        result = self.controller.send_statustext_msg(message)
        if not result:
            self.__logger.error("Failed to send statustext message", True)

        return True, odometry_and_time_object

    def apply_decision(self, cmd: decision_command.DecisionCommand) -> bool:
        """
        Applies the decision command to the drone.
        Returns True if successful, False otherwise.
        """
        command_type = cmd.get_command_type()
        command_position = cmd.get_command_position()

        if command_type == decision_command.DecisionCommand.CommandType.MOVE_TO_RELATIVE_POSITION:
            # Move relative to current position.
            # Get current position.
            result, current_odometry = self.controller.get_odometry()
            if not result or current_odometry is None:
                return False

            # Convert current global position to local NED coordinates.
            result, current_local_odometry = (
                local_global_conversion.drone_odometry_local_from_global(
                    self.__home_position, current_odometry
                )
            )
            if not result or current_local_odometry is None:
                return False

            # Add relative offsets.
            target_north = current_local_odometry.position.north + command_position[0]
            target_east = current_local_odometry.position.east + command_position[1]
            target_down = current_local_odometry.position.down + command_position[2]

            result, target_local_position = position_local.PositionLocal.create(
                target_north, target_east, target_down
            )
            if not result or target_local_position is None:
                return False

            result, target_global_position = (
                local_global_conversion.position_global_from_position_local(
                    self.__home_position, target_local_position
                )
            )
            if not result or target_global_position is None:
                return False

            # Move to target global position.
            return self.controller.move_to_position(target_global_position)

        if command_type == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION:
            # Move to absolute position.
            # Note that command_position[2] is the absolute altitude not relative altitude.
            result, target_position = position_global.PositionGlobal.create(
                command_position[0], command_position[1], command_position[2]
            )

            if not result or target_position is None:
                return False

            return self.controller.move_to_position(target_position)

        if command_type == decision_command.DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION:
            # Simply switch flight mode to LAND.
            return self.controller.set_flight_mode("LAND")

        if command_type == decision_command.DecisionCommand.CommandType.LAND_AT_RELATIVE_POSITION:
            # Land at relative position.
            # Get current position.
            result, current_odometry = self.controller.get_odometry()
            if not result or current_odometry is None:
                return False

            # Convert current global position to local NED coordinates
            result, current_local_odometry = (
                local_global_conversion.drone_odometry_local_from_global(
                    self.__home_position, current_odometry
                )
            )
            if not result or current_local_odometry is None:
                return False

            # Add relative offsets.
            target_north = current_local_odometry.position.north + command_position[0]
            target_east = current_local_odometry.position.east + command_position[1]
            target_down = current_local_odometry.position.down + command_position[2]

            # Create target local position.
            result, target_local_position = position_local.PositionLocal.create(
                target_north, target_east, target_down
            )
            if not result or target_local_position is None:
                return False
            # Convert target local position to global position.
            result, target_global_position = (
                local_global_conversion.position_global_from_position_local(
                    self.__home_position, target_local_position
                )
            )
            if not result or target_global_position is None:
                return False
            # Upload land command.
            result = self.controller.upload_land_command(
                target_global_position.latitude, target_global_position.longitude
            )
            if not result:
                return False

            return self.controller.set_flight_mode("AUTO")

        if command_type == decision_command.DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION:
            # Land at absolute position in local NED coordinates
            result = self.controller.upload_land_command(command_position[0], command_position[1])
            if not result:
                return False

            return self.controller.set_flight_mode("AUTO")

        # Unsupported commands
        return False
