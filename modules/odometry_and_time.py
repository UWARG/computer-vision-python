"""
<<<<<<< HEAD
Drone odometry in local space and timestamp.
=======
Drone odometry and timestamp
>>>>>>> 78c49d9 (Update odometry_and_time.py)
"""
import time

from . import drone_odometry_local


# Basically a struct
# pylint: disable=too-few-public-methods
class OdometryAndTime:
    """
    Contains odometry/telemetry and timestamp
    """
    __create_key = object()

    @classmethod
<<<<<<< HEAD
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
=======
    def create(cls, 
               position: drone_odometry.DronePosition, 
               orientation: drone_odometry.DroneOrientation) -> "tuple[bool, MergedOdometryDetections | None]":
        """
        position: Latitude, longitude in decimal degrees and altitude in metres
        orientation: Yaw, pitch, roll following NED system (x forward, y right, z down)
        """

        return True, OdometryAndTime(cls.__create_key, position, orientation)
        
    def __init__(self,
                 class_private_create_key,
                 position: drone_odometry.DronePosition,
                 orientation: drone_odometry.DroneOrientation):
        """
        Private constructor, use create() method
        Constructor sets timestamp to current time
        """
        assert class_private_create_key is OdometryAndTime.__create_key, "Use create() method"
                     
        self.position = position
        self.orientation = orientation
        self.timestamp = time.time()
>>>>>>> 78c49d9 (Update odometry_and_time.py)

# pylint: enable=too-few-public-methods
