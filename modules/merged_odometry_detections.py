"""
Drone odometry and object detections
"""

from . import detections_and_time
<<<<<<< HEAD
from . import drone_odometry_local
=======
from . import odometry_and_time
>>>>>>> bd110f8 (Update merged_odometry_detections.py)


# Basically a struct
# pylint: disable=too-few-public-methods
class MergedOdometryDetections:
    """
    Contains odometry/telemetry and detections merged by closest timestamp
    """
    __create_key = object()

    @classmethod
    def create(cls, drone_position, drone_orientation, detections) -> "tuple[bool, MergedOdometryDetections | None]":
        """
        detections: list of Detections from detections_and_time
        """
        if len(detections) == 0:
            return False, None

        return True, MergedOdometryDetections(cls.__create_key, drone_position, drone_orientation, detections)
    
    def __init__(self,
<<<<<<< HEAD
                 odometry_local: drone_odometry_local.DroneOdometryLocal,
=======
                 class_private_create_key,
                 drone_position: odometry_and_time.DronePosition,
                 drone_orientation: odometry_and_time.DroneOrientation,
>>>>>>> bd110f8 (Update merged_odometry_detections.py)
                 detections: "list[detections_and_time.Detection]"):
        """
        Private constructor, use create() method
        Required for separation
        """
<<<<<<< HEAD
        self.odometry_local = odometry_local
=======
        assert class_private_create_key is MergedOdometryDetections.__create_key, "Use create() method"

        self.drone_position = drone_position
        self.drone_orientation = drone_orientation
>>>>>>> bd110f8 (Update merged_odometry_detections.py)
        self.detections = detections

# pylint: enable=too-few-public-methods
