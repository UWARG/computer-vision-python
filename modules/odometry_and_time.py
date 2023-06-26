"""
Drone odometry and timestamp
"""
import math
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class PositionWorld:
    """
    WGS 84 following ISO 6709 (latitude before longitude)
    """
    __create_key = object()

    @classmethod
    def create(cls, latitude: float, longitude: float, altitude: float) -> "tuple[bool, PositionWorld | None]":
        """
        latitude, longitude in decimal degrees
        altitude in metres
        """
        if altitude <= 0.0:
            return False, None

        return True, PositionWorld(cls.__create_key, latitude, longitude, altitude)

    def __init__(self, class_private_create_key, latitude: float, longitude: float, altitude: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is PositionWorld.__create_key, "Use create() method"

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class OrientationWorld:
    """
    Yaw, pitch, roll following NED system (x forward, y right, z down)
    Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order
    """
    __create_key = object()

    @classmethod
    def create(cls, yaw: float, pitch: float, roll: float) -> "tuple[bool, OrientationWorld | None]":
        """
        yaw, pitch, roll in radians
        """
        if yaw < -math.pi or yaw > math.pi:
            return False, None

        if pitch < -math.pi or pitch > math.pi:
            return False, None

        if roll < -math.pi or roll > math.pi:
            return False, None

        return True, OrientationWorld(cls.__create_key, yaw, pitch, roll)

    def __init__(self, class_private_create_key, yaw: float, pitch: float, roll: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is OrientationWorld.__create_key, "Use create() method"

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
    def __init__(self, position: PositionWorld, orientation: OrientationWorld):
        """
        Constructor sets timestamp to current time
        message: 1 of TelemMessages. No type annotation due to several possible types
        """
        self.position = position
        self.orientation = orientation
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
