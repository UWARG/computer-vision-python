"""
Creates flight controller and combines odometry data and timestamp.
"""

import inspect
import dronekit

from . import local_global_conversion
from .. import odometry_and_time
from .. import decision_command
from ..logger import logger
from ..common.mavlink.modules import drone_odometry
from ..common.mavlink.modules import flight_controller


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
            frame = inspect.currentframe()
            local_logger.error("controller could not be created", frame)
            return False, None
        # Get Pylance to stop complaining

        assert controller is not None

        result, home_location = controller.get_home_location(timeout_home)
        if not result:
            frame = inspect.currentframe()
            local_logger.error("home_location could not be created", frame)
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

        frame = inspect.currentframe()
        self.__logger.info(str(self.__home_location), frame)

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

    def apply_decision(self, decision: decision_command.DecisionCommand) -> None:
        """
        Apply the given decision command to the flight controller.
        """
        commands = self._convert_decision_to_commands(decision)
        self.controller.upload_commands(commands)

    def _convert_decision_to_commands(
        self, decision: decision_command.DecisionCommand
    ) -> "list[dronekit.Command]":
        """
        Converts a DecisionCommand into a list of dronekit.Command objects.
        """
        commands = []

        # Extract the command type and position data from the DecisionCommand
        command_type = decision.get_command_type()
        x, y, z = decision.get_command_position()

        if command_type == decision_command.DecisionCommand.CommandType.MOVE_TO_RELATIVE_POSITION:
            # Convert relative position to absolute position
            # Assuming you have the current position available
            current_position = self.controller.get_current_position()
            target_latitude = current_position.latitude + x
            target_longitude = current_position.longitude + y
            target_altitude = current_position.altitude + z

            # Create a NAV_WAYPOINT command to move to the target position
            move_command = dronekit.Command(
                0,
                0,
                0,
                dronekit.mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                dronekit.mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,
                0,
                0,
                0,
                0,
                0,
                target_latitude,
                target_longitude,
                target_altitude,
            )
            commands.append(move_command)

        elif command_type == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION:
            # Create a NAV_WAYPOINT command to move to the target absolute position
            move_command = dronekit.Command(
                0,
                0,
                0,
                dronekit.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                dronekit.mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,
                0,
                0,
                0,
                0,
                0,
                x,
                y,
                z,
            )
            commands.append(move_command)

        elif command_type == decision_command.DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION:
            # Create a LAND command at the current position
            current_position = self.controller.get_current_position()
            land_command = dronekit.Command(
                0,
                0,
                0,
                dronekit.mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                dronekit.mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                current_position.latitude,
                current_position.longitude,
                current_position.altitude,
            )
            commands.append(land_command)

        elif command_type == decision_command.DecisionCommand.CommandType.LAND_AT_RELATIVE_POSITION:
            # Convert relative position to absolute position
            current_position = self.controller.get_current_position()
            target_latitude = current_position.latitude + x
            target_longitude = current_position.longitude + y
            target_altitude = current_position.altitude + z

            # Create a LAND command at the target position
            land_command = dronekit.Command(
                0,
                0,
                0,
                dronekit.mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                dronekit.mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                target_latitude,
                target_longitude,
                target_altitude,
            )
            commands.append(land_command)

        elif command_type == decision_command.DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION:
            # Create a LAND command at the absolute position
            land_command = dronekit.Command(
                0,
                0,
                0,
                dronekit.mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                dronekit.mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                x,
                y,
                z,
            )
            commands.append(land_command)

        else:
            raise ValueError(f"Unsupported command type: {command_type}")

        return commands
