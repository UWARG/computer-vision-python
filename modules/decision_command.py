import enum


class Command:
    """
    Command struct for the decisions module.
    The following constructors are available for different command types:

    * Command.create_relative_movement_command()
    * Command.create_absolute_movement_command()
    * Command.create_land_immediate_command()
    * Command.create_land_at_position_command()

    """
    class CommandType(enum.Enum):
        """
        Different types of commands.
        """
        MOVE_RELATIVE    = 0  # Move relative to current position
        MOVE_ABSOLUTE    = 1  # Move to absolute position within local space
        LAND_IMMEDIATE   = 2  # Stop the drone at current position
        LAND_AT_POSITION = 3  # Stop the drone at position within local space

    __create_key = object()

    @classmethod
    def create_relative_movement_command(cls,
                                         relative_x: float,
                                         relative_y: float,
                                         relative_z: float) -> "Command":
        """
        Command for drone movement relative to current position.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.MOVE_RELATIVE,
            relative_x,
            relative_y,
            relative_z
        )

    @classmethod
    def create_absolute_movement_command(cls,
                                        absolute_x: float,
                                        absolute_y: float,
                                        absolute_z: float) -> "Command":
        """
        Command for drone movement to absolute position within local space.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.MOVE_ABSOLUTE,
            absolute_x,
            absolute_y,
            absolute_z
        )

    @classmethod
    def create_land_immediate_command(cls) -> "Command":
        """
        Command for landing at current position.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.LAND_IMMEDIATE,
            0.0,
            0.0,
            0.0
        )

    @classmethod
    def create_land_at_position_command(cls, 
                                        x: float, 
                                        y: float,
                                        z: float) -> "Command":
        """
        Command to land the drone at an absolute position within local space.
        """
        return Command(
            cls.__create_key,
            Command.CommandType.LAND_AT_POSITION,
            x,
            y,
            z
        )

    def __init__(self,
                 class_private_create_key,
                 command_type: CommandType,
                 command_x: float,
                 command_y: float,
                 command_z: float):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Command.__create_key, "Use create() method"

        self.__command_type = command_type
        self.__command_x = command_x
        self.__command_y = command_y
        self.__command_z = command_z

    def get_command_type(self) -> CommandType:
        """
        Getter.
        """
        return self.__command_type

    def get_command_position(self) -> "tuple[float, float, float]":
        """
        Getter.
        """
        return self.__command_x, self.__command_y, self.__command_z

    def __repr__(self) -> str:
        """
        To string.
        """
        representation = "Command: " + str(self.__command_type)

        if self.__command_type != Command.CommandType.LAND_IMMEDIATE:
            representation += " " + str(self.get_command_position())

        return representation
