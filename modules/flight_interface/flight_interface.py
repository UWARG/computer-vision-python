"""
Creates flight controller and combines odometry data and timestamp.
"""

from .. import decision_command, odometry_and_time
from ..common.logger.modules import logger
from ..common.mavlink.modules import drone_odometry, drone_odometry_local, flight_controller, local_global_conversion  # fmt: skip


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

        result, home_location = controller.get_home_location(timeout_home)
        if not result:
            local_logger.error("home_location could not be created", True)
            return False, None

        # Get Pylance to stop complaining
        assert home_location is not None

        return True, FlightInterface(cls.__create_key, controller, home_location, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        controller: flight_controller.FlightController,
        home_location: drone_odometry.DronePosition,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller
        self.__home_location = home_location
        self.__logger = local_logger

        self.__logger.info(str(self.__home_location), True)

    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = self.controller.get_odometry()
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        result, odometry_local = local_global_conversion.drone_odometry_local_from_global(
            odometry,
            self.__home_location,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_local is not None
        return odometry_and_time.OdometryAndTime.create(odometry_local)

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
                    current_odometry, self.__home_location
                )
            )
            if not result or current_local_odometry is None:
                return False

            # Add relative offsets.
            target_north = current_local_odometry.position.north + command_position[0]
            target_east = current_local_odometry.position.east + command_position[1]
            target_down = current_local_odometry.position.down + command_position[2]

            result, target_local_position = drone_odometry_local.DronePositionLocal.create(
                target_north, target_east, target_down
            )
            if not result or target_local_position is None:
                return False

            result, target_global_position = (
                local_global_conversion.drone_position_global_from_local(
                    self.__home_location, target_local_position
                )
            )
            if not result or target_global_position is None:
                return False

            # Move to target global position.
            return self.controller.move_to_position(target_global_position)

        if command_type == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION:
            # Move to absolute position.
            # Note that command_position[2] is the absolute altitude not relative altitude.
            result, target_position = drone_odometry.DronePosition.create(
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
                    current_odometry, self.__home_location
                )
            )
            if not result or current_local_odometry is None:
                return False

            # Add relative offsets.
            target_north = current_local_odometry.position.north + command_position[0]
            target_east = current_local_odometry.position.east + command_position[1]
            target_down = current_local_odometry.position.down + command_position[2]

            # Create target local position.
            result, target_local_position = drone_odometry_local.DronePositionLocal.create(
                target_north, target_east, target_down
            )
            if not result or target_local_position is None:
                return False
            # Convert target local position to global position.
            result, target_global_position = (
                local_global_conversion.drone_position_global_from_local(
                    self.__home_location, target_local_position
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
