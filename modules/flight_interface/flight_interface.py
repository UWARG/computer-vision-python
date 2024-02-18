"""
Creates flight controller and combines odometry data and timestamp.
"""

from . import local_global_conversion
from .. import odometry_and_time
from ..common.mavlink.modules import drone_odometry
from ..common.mavlink.modules import flight_controller


# This is just an interface
# pylint: disable-next=too-few-public-methods
class FlightInterface:
    """
    Create flight controller and combines odometry data and timestamp.
    """

    __create_key = object()

    @classmethod
    def create(cls, address: str, timeout_home: float) -> "tuple[bool, FlightInterface | None]":
        """
        address: TCP address or port.
        timeout_home: Timeout for home location in seconds.
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

    def __init__(
        self,
        class_private_create_key,
        controller: flight_controller.FlightController,
        home_location: drone_odometry.DronePosition,
    ):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller
        self.__home_location = home_location

    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = self.controller.get_odometry()
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        result, odometry_local = local_global_conversion.drone_odometry_local_from_global(
            odometry,
            self.__home_location,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_local is not None

        return odometry_and_time.OdometryAndTime.create(odometry_local)
