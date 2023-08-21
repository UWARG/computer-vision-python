"""
Drone odometry and object detections.
"""

from . import detections_and_time
from .common.mavlink.modules import drone_odometry


# Basically a struct
# pylint: disable=too-few-public-methods
class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp.
    """
    def __init__(self,
                 drone_position: drone_odometry.DronePosition,
                 drone_orientation: drone_odometry.DroneOrientation,
                 detections: "list[detections_and_time.Detection]"):
        """
        Required for separation.
        """
        self.drone_position = drone_position
        self.drone_orientation = drone_orientation
        self.detections = detections

# pylint: enable=too-few-public-methods
