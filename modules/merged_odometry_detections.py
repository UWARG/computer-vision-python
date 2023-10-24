"""
Drone odometry and object detections.
"""

from . import detections_and_time
from . import drone_odometry_local


# Basically a struct
# pylint: disable=too-few-public-methods
class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp.
    """
    def __init__(self,
                 odometry_local: drone_odometry_local.DroneOdometryLocal,
                 detections: "list[detections_and_time.Detection]"):
        """
        Required for separation.
        """
        self.odometry_local = odometry_local
        self.detections = detections

# pylint: enable=too-few-public-methods
