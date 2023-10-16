"""
Creates flight controller and combines odometry data and timestamp.
"""

import pymap3d as pm

from .. import drone_odometry_local
from .. import odometry_and_time
from ..common.mavlink.modules import drone_odometry
from ..common.mavlink.modules import flight_controller

# This is just an interface
# pylint: disable=too-few-public-methods
class FlightInterface:
    """
    Create flight controller and combines odometry data and timestamp.
    """
    __create_key = object()

    @classmethod
    def create(cls, address: str, timeout_home: float) -> "tuple[bool, FlightInterface | None]":
        """
        address: TCP or port.
        timeout: Timeout for home location in seconds.
        """
        result, controller = flight_controller.FlightController.create(address)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert controller is not None

        result, home_location = controller.get_home_location(timeout_home)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert home_location is not None

        return True, FlightInterface(cls.__create_key, controller, home_location)

    def __init__(self,
                 class_private_create_key,
                 controller: flight_controller.FlightController,
                 home_location: drone_odometry.DronePosition):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller
        self.__home_location = home_location

    @staticmethod
    def __drone_position_global_from_local(home_location: drone_odometry.DronePosition,
                                           drone_position_local:
                                               drone_odometry_local.DronePositionLocal) \
        -> "tuple[bool, drone_odometry.DronePosition | None]":
        """
        Local coordinates to global coordinates.
        Return: Drone position in WGS 84.
        """
        latitude, longitude, altitude = pm.ned2geodetic(
            drone_position_local.north,
            drone_position_local.east,
            drone_position_local.down,
            home_location.latitude,
            home_location.longitude,
            home_location.altitude,
        )

        result, drone_position = drone_odometry.DronePosition.create(
            latitude,
            longitude,
            altitude,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert drone_position is not None

        return True, drone_position

    @staticmethod
    def __drone_position_local_from_global(home_location: drone_odometry.DronePosition,
                                           drone_position: drone_odometry.DronePosition) \
        -> "tuple[bool, drone_odometry_local.DronePositionLocal | None]":
        """
        Global coordinates to local coordinates.
        Return: Drone position relative to home location (NED system).
        """
        north, east, down = pm.geodetic2ned(
            drone_position.latitude,
            drone_position.longitude,
            drone_position.altitude,
            home_location.latitude,
            home_location.longitude,
            home_location.altitude,
        )

        result, drone_position_local = drone_odometry_local.DronePositionLocal.create(
            north,
            east,
            down,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert drone_position_local is not None

        return True, drone_position_local

    @staticmethod
    def __drone_odometry_local_from_global(odometry: drone_odometry.DroneOdometry,
                                           home_location: drone_odometry.DronePosition) \
        -> "tuple[bool, drone_odometry_local.DroneOdometryLocal | None]":
        """
        Converts global odometry to local.
        """
        result, drone_position_local = FlightInterface.__drone_position_local_from_global(
            home_location,
            odometry.position,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert drone_position_local is not None

        result, drone_orientation_local = drone_odometry_local.DroneOrientationLocal.create_wrap(
            odometry.orientation,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert drone_orientation_local is not None

        return drone_odometry_local.DroneOdometryLocal.create(
            drone_position_local,
            drone_orientation_local,
        )

    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = self.controller.get_odometry()
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        result, odometry_local = self.__drone_odometry_local_from_global(
            odometry,
            self.__home_location,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_local is not None

        return odometry_and_time.OdometryAndTime.create(odometry_local)

# pylint: enable=too-few-public-methods
