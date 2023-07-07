"""
Drone odometry and timestamp
"""
import math
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class DronePosition:
    """
    WGS 84 following ISO 6709 (latitude before longitude)
    """
    __create_key = object()

    @classmethod
    def create(cls, latitude: float, longitude: float, altitude: float) -> "tuple[bool, DronePosition | None]":
        """
        latitude, longitude in decimal degrees
        altitude in metres
        """
        if altitude <= 0.0:
            return False, None

        return True, DronePosition(cls.__create_key, latitude, longitude, altitude)

    def __init__(self, class_private_create_key, latitude: float, longitude: float, altitude: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is DronePosition.__create_key, "Use create() method"

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class DroneOrientation:
    """
    Yaw, pitch, roll following NED system (x forward, y right, z down)
    Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order
    """
    __create_key = object()

    @classmethod
    def create(cls, yaw: float, pitch: float, roll: float) -> "tuple[bool, DroneOrientation | None]":
        """
        yaw, pitch, roll in radians
        """
        if yaw < -math.pi or yaw > math.pi:
            return False, None

        if pitch < -math.pi or pitch > math.pi:
            return False, None

        if roll < -math.pi or roll > math.pi:
            return False, None

        return True, DroneOrientation(cls.__create_key, yaw, pitch, roll)

    def __init__(self, class_private_create_key, yaw: float, pitch: float, roll: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is DroneOrientation.__create_key, "Use create() method"

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class OdometryAndTime:
    """
    Contains odometry/telemetry and timestamp
    """
    def __init__(self, position: DronePosition, orientation: DroneOrientation):
        """
        Constructor sets timestamp to current time
        """
        self.position = position
        self.orientation = orientation
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
