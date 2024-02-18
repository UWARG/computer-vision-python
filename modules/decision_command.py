"""
Commands for the decision module.
"""

import enum


class DecisionCommand:
    """
    Contains command type and coordinate data.

    All coordinate values use the NED coordinate system. Positive x is north,
    positive y is east, positive z is down.

    The following constructors are available for different command types:

    * Command.create_move_to_relative_position_command
    * Command.create_move_to_absolute_position_command
    * Command.create_land_at_current_position_command
    * Command.create_land_at_relative_position_command
    * Command.create_land_at_absolute_position_command

    """

    __create_key = object()

    class CommandType(enum.Enum):
        """
        Different types of commands.
        """

        MOVE_TO_RELATIVE_POSITION = 0  # Move relative to current position
        MOVE_TO_ABSOLUTE_POSITION = 1  # Move to absolute position within local space
        LAND_AT_CURRENT_POSITION = 2  # Stop the drone at current position
        LAND_AT_RELATIVE_POSITION = 3  # Stop the drone at relative position within local space
        LAND_AT_ABSOLUTE_POSITION = 4  # Stop the drone at absolute position within local space

    @classmethod
    def create_move_to_relative_position_command(
        cls, relative_x: float, relative_y: float, relative_z: float
    ) -> "DecisionCommand":
        """
        Command for drone movement relative to current position, using
        the NED coordinate system. (+x, +y, +z) corresponds to the north, east, and down directions.
        """
        return DecisionCommand(
            cls.__create_key,
            DecisionCommand.CommandType.MOVE_TO_RELATIVE_POSITION,
            relative_x,
            relative_y,
            relative_z,
        )

    @classmethod
    def create_move_to_absolute_position_command(
        cls, absolute_x: float, absolute_y: float, absolute_z: float
    ) -> "DecisionCommand":
        """
        Command for drone movement to absolute position within local space, using
        the NED coordinate system. (+x, +y, +z) corresponds to the north, east, and down directions.
        """
        return DecisionCommand(
            cls.__create_key,
            DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION,
            absolute_x,
            absolute_y,
            absolute_z,
        )

    @classmethod
    def create_land_at_current_position_command(cls) -> "DecisionCommand":
        """
        Command for landing at current position.
        """
        return DecisionCommand(
            cls.__create_key, DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION, 0.0, 0.0, 0.0
        )

    @classmethod
    def create_land_at_relative_position_command(
        cls, relative_x: float, relative_y: float, relative_z: float
    ) -> "DecisionCommand":
        """
        Command to land the drone at a relative position within local space, using
        the NED coordinate system. (+x, +y, +z) corresponds to the north, east, and down directions.
        """
        return DecisionCommand(
            cls.__create_key,
            DecisionCommand.CommandType.LAND_AT_RELATIVE_POSITION,
            relative_x,
            relative_y,
            relative_z,
        )

    @classmethod
    def create_land_at_absolute_position_command(
        cls, absolute_x: float, absolute_y: float, absolute_z: float
    ) -> "DecisionCommand":
        """
        Command to land the drone at an absolute position within local space, using
        the NED coordinate system. (+x, +y, +z) corresponds to the north, east, and down directions.
        """
        return DecisionCommand(
            cls.__create_key,
            DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION,
            absolute_x,
            absolute_y,
            absolute_z,
        )

    def __init__(
        self,
        class_private_create_key,
        command_type: CommandType,
        command_x: float,
        command_y: float,
        command_z: float,
    ):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DecisionCommand.__create_key, "Use create() method"

        self.__command_type = command_type
        self.__command_x = command_x
        self.__command_y = command_y
        self.__command_z = command_z

    def get_command_type(self) -> CommandType:
        """
        Returns the command type enum.
        """
        return self.__command_type

    def get_command_position(self) -> "tuple[float, float, float]":
        """
        Returns the command position in x, y, z tuple, using
        the NED coordinate system. (+x, +y, +z) corresponds to the north, east, and down directions.
        """
        return self.__command_x, self.__command_y, self.__command_z

    def __repr__(self) -> str:
        """
        To string.
        """
        representation = "Command: " + str(self.__command_type)

        if self.__command_type != DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION:
            representation += " " + str(self.get_command_position())

        return representation
