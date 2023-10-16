"""
Drone odometry in local space and timestamp.
"""
import time

from . import drone_odometry_local


# Basically a struct
# pylint: disable=too-few-public-methods
class OdometryAndTime:
    """
    Contains odometry/telemetry and timestamp.
    """
    __create_key = object()

    @classmethod
    def create(cls, odometry_data: drone_odometry_local.DroneOdometryLocal) \
        -> "tuple[bool, OdometryAndTime | None]":
        """
        Timestamps the odometry with the current time.
        odometry_data: Drone odometry data.
        """
        if odometry_data is None:
            return False, None

        timestamp = time.time()

        return True, OdometryAndTime(cls.__create_key, odometry_data, timestamp)

    def __init__(self,
                 class_private_create_key,
                 odometry_data: drone_odometry_local.DroneOdometryLocal,
                 timestamp: float):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is OdometryAndTime.__create_key, "Use create() method"

        self.odometry_data = odometry_data
        self.timestamp = timestamp

# pylint: enable=too-few-public-methods
