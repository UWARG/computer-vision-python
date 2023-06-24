"""
Drone odometry and timestamp
"""
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class PositionWorld:
    """
    WGS 84 following ISO 6709
    """
    def __init__(self, latitude: float, longitude: float, altitude: float):
        assert altitude >= 0

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class OrientationWorld:
    """
    Yaw Pitch Roll following NED system (x forward, y right, z down)
    Specifically, intrinsic (Tait-Bryan) rotations in the zyx/3-2-1 order
    """
    def __init__(self, yaw: float, pitch: float, roll: float):
        # TODO: Verify bounds
        assert yaw >= -180.0
        assert yaw <= 180.0
        assert pitch >= -180.0
        assert pitch <= 180.0
        assert roll >= -180.0
        assert roll <= 180.0

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
