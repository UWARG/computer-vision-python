"""
Drone odometry in local space (origin at home location).
"""

from .common.mavlink.modules import drone_odometry


class DronePositionLocal:
    """
    Drone position in NED system.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, north: float, east: float, down: float
    ) -> "tuple[bool, DronePositionLocal | None]":
        """
        North, east, down in metres.
        """
        return True, DronePositionLocal(cls.__create_key, north, east, down)

    def __init__(
        self, class_private_create_key: object, north: float, east: float, down: float
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DronePositionLocal.__create_key, "Use create() method"

        self.north = north
        self.east = east
        self.down = down

    def __str__(self) -> str:
        """
        To string.
        """
        return f"DronePositionLocal (NED): {self.north}, {self.east}, {self.down}"


class DroneOrientationLocal:
    """
    Wrapper for DroneOrientation as it is the same in both local and global space.
    """

    __create_key = object()

    @classmethod
    def create_new(
        cls, yaw: float, pitch: float, roll: float
    ) -> "tuple[bool, DroneOrientationLocal | None]":
        """
        Yaw, pitch, roll in radians.
        """
        result, orientation = drone_odometry.DroneOrientation.create(yaw, pitch, roll)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert orientation is not None

        return True, DroneOrientationLocal(cls.__create_key, orientation)

    @classmethod
    def create_wrap(
        cls, orientation: drone_odometry.DroneOrientation
    ) -> "tuple[bool, DroneOrientationLocal | None]":
        """
        Wrap existing orientation.
        """
        return True, DroneOrientationLocal(cls.__create_key, orientation)

    def __init__(
        self, class_private_create_key: object, orientation: drone_odometry.DroneOrientation
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DroneOrientationLocal.__create_key, "Use create() method"

        self.orientation = orientation

    def __str__(self) -> str:
        """
        To string.
        """
        # TODO: Update common
        return f"DroneOrientationLocal (YPR rad): {self.orientation.yaw}, {self.orientation.pitch}, {self.orientation.roll}"


class DroneOdometryLocal:
    """
    Wrapper for DronePositionLocal and DroneOrientationLocal.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, position: DronePositionLocal, orientation: DroneOrientationLocal
    ) -> "tuple[bool, DroneOdometryLocal | None]":
        """
        Position and orientation in one class.
        """
        if position is None:
            return False, None

        if orientation is None:
            return False, None

        return True, DroneOdometryLocal(cls.__create_key, position, orientation)

    def __init__(
        self,
        class_private_create_key: object,
        position: DronePositionLocal,
        orientation: DroneOrientationLocal,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DroneOdometryLocal.__create_key, "Use create() method"

        self.position = position
        self.orientation = orientation

    def __str__(self) -> str:
        """
        To string.
        """
        return f"DroneOdometryLocal: {self.position}, {self.orientation}"
