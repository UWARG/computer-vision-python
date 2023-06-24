"""
Drone odometry and object detections
"""

from modules import odometry_and_time
from . import detections_and_time


# Basically a struct
# pylint: disable=too-few-public-methods
class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp
    """
    def __init__(self,
                 position: odometry_and_time.PositionWorld,
                 orientation: odometry_and_time.OrientationWorld,
                 detections: "list[detections_and_time.Detection]"):
        """
        Required for separation
        """
        self.position = position
        self.orientation = orientation
        self.detections = detections

# pylint: enable=too-few-public-methods
