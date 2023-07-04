"""
Drone odometry and object detections
"""

from . import detections_and_time
from . import odometry_and_time


# Basically a struct
# pylint: disable=too-few-public-methods
class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp
    """
    def __init__(self,
                 drone_position: odometry_and_time.DronePosition,
                 drone_orientation: odometry_and_time.DroneOrientation,
                 detections: "list[detections_and_time.Detection]"):
        """
        Required for separation
        """
        self.drone_position = drone_position
        self.drone_orientation = drone_orientation
        self.detections = detections

# pylint: enable=too-few-public-methods
