"""
Drone odometry and timestamp.
"""
import time

from .common.mavlink.modules import drone_odometry


# Basically a struct
# pylint: disable=too-few-public-methods
class OdometryAndTime:
    """
    Contains odometry/telemetry and timestamp.
    """
    __create_key = object()

    @classmethod
    def create(cls, odometry_data: drone_odometry.DroneOdometry) -> "tuple[bool, OdometryAndTime | None]":
        """
        odometry_data: Mavlink/Dronekit odometry data from drone_odometry class.
        """
        if odometry_data is None:
            return False, None
        
        assert cls.__create_key is OdometryAndTime.__create_key, "Use create method"
        
        return True, OdometryAndTime(odometry_data)

    def __init__(self, odometry_data: drone_odometry.DroneOdometry):
        """
        Private constructor, use create() method.
        Constructor sets timestamp to current time.
        """
        
        
        self.odometry_data = odometry_data
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
