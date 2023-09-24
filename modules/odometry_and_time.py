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
        Private constructor, use create() method.
        odometry_data: Mavlink/Dronekit odometry data from drone_odometry class.
        """
        
        return True, OdometryAndTime(cls.__create_key, odometry_data)

    def __init__(self, class_private_create_key, odometry_data: drone_odometry.DroneOdometry):
        """
        Constructor sets timestamp to current time.
        """
        if odometry_data is None:
            return False, None
        
        assert class_private_create_key is OdometryAndTime.__create_key, "Use create() method"
        
        self.odometry_data = odometry_data
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
