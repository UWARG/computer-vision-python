"""
Drone odometry in local space (origin at home location).
"""

from .common.mavlink.modules import drone_odometry


# Basically a struct
# pylint: disable=too-few-public-methods
class DronePositionLocal:
    """
    Drone position in NED system.
    """
    __create_key = object()

    @classmethod
    def create(cls,
               north: float,
               east: float,
               down: float) -> "tuple[bool, DronePositionLocal | None]":
        """
        north, east, down in metres.
        """
        return True, DronePositionLocal(cls.__create_key, north, east, down)

    def __init__(self,
                 class_private_create_key,
                 north: float,
                 east: float,
                 down: float):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DronePositionLocal.__create_key, "Use create() method"

        self.north = north
        self.east = east
        self.down = down

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class DroneOrientationLocal:
    """
    Wrapper for DroneOrientation as it is the same in both local and global space.
    """
    __create_key = object()

    @classmethod
    def create_new(cls,
                   yaw: float,
                   pitch: float,
                   roll: float) -> "tuple[bool, DroneOrientationLocal | None]":
        """
        yaw, pitch, roll in radians.
        """
        result, orientation = drone_odometry.DroneOrientation.create(yaw, pitch, roll)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert orientation is not None

        return True, DroneOrientationLocal(cls.__create_key, orientation)

    @classmethod
    def create_wrap(cls,
                    orientation: drone_odometry.DroneOrientation):
        """
        Wrap existing orientation.
        """
        return True, DroneOrientationLocal(cls.__create_key, orientation)

    def __init__(self, class_private_create_key, orientation: drone_odometry.DroneOrientation):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DroneOrientationLocal.__create_key, "Use create() method"

        self.orientation = orientation

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class DroneOdometryLocal:
    """
    Wrapper for DronePositionLocal and DroneOrientationLocal.
    """
    __create_key = object()

    @classmethod
    def create(cls,
               position: DronePositionLocal,
               orientation: DroneOrientationLocal) -> "tuple[bool, DroneOdometryLocal | None]":
        """
        Position and orientation in one class.
        """
        if position is None:
            return False, None

        if orientation is None:
            return False, None

        return True, DroneOdometryLocal(cls.__create_key, position, orientation)

    def __init__(self,
                 class_private_create_key,
                 position: DronePositionLocal,
                 orientation: DroneOrientationLocal):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DroneOdometryLocal.__create_key, "Use create() method"

        self.position = position
        self.orientation = orientation
