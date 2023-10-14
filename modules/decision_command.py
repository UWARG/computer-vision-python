import enum


class Command:
    """
    Command struct for the decisions module.
    The following constructors are available for different command types:

    """
    class CommandType(enum.Enum):
        """
        Different types of commands.
        """
        MOVE_RELATIVE    = 0  # Move relative to current position
        MOVE_ABSOLUTE    = 1  # Move to absolute position
        LAND_IMMEDIATE   = 2  # Stop the drone at current position
        LAND_AT_POSITION = 3  # Stop the drone at position

    __create_key = object()

    @classmethod
    def create_relative_movement_command(cls,
                                         relative_x: float,
                                         relative_y: float) -> "Command":
        """
        Command for drone movement relative to current position.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.MOVE_RELATIVE,
            relative_x,
            relative_y,
        )

    @classmethod
    def create_absolute_movement_command(cls,
                                        absolute_x: float,
                                        absolute_y: float) -> "Command":
        """
        Command to set drone destination.
        Drone must be in halted state.
        relative is distance from current position of the drone in metres.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.MOVE_ABSOLUTE,
            absolute_x,
            absolute_y,
        )

    @classmethod
    def create_land_immediate_command(cls) -> "Command":
        """
        Command to halt the drone.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.LAND_IMMEDIATE,
            0.0,
            0.0,
        )

    @classmethod
    def create_land_at_position_command(cls, 
                                        x: float, 
                                        y: float) -> "Command":
        """
        Command to land the drone.
        Drone must be in halted state.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.LAND_AT_POSITION,
            x,
            y,
        )

    def __init__(self,
                 class_private_create_key,
                 command_type: CommandType,
                 command_x: float,
                 command_y: float):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Command.__create_key, "Use create() method"

        self.__command_type = command_type
        self.__command_x = command_x
        self.__command_y = command_y

    def get_command_type(self) -> CommandType:
        """
        Getter.
        """
        return self.__command_type

    def get_command_position(self) -> "tuple[float, float]":
        """
        Getter.
        """
        return self.__command_x, self.__command_y

    def __repr__(self) -> str:
        """
        To string.
        """
        representation = "Command: " + str(self.__command_type)

        if self.__command_type != Command.CommandType.LAND_IMMEDIATE:
            representation += " " + str(self.get_command_position())

        return representation
