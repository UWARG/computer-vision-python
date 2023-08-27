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
    def __init__(self,
                 position: drone_odometry.DronePosition,
                 orientation: drone_odometry.DroneOrientation):
        """
        Constructor sets timestamp to current time.
        """
        self.position = position
        self.orientation = orientation
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
