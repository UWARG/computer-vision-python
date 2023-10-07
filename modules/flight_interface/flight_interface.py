"""
Creates flight controller and combines odometry data and timestamp.
"""

from .. import odometry_and_time
from ..common.mavlink.modules import flight_controller


# This is just an interface
# pylint: disable=too-few-public-methods
class FlightInterface:
    """
    Create flight controller and combines odometry data and timestamp.
    """
    __create_key = object()

    @classmethod
    def create(cls, address: str) -> "tuple[bool, FlightInterface | None]":
        """
        address: TCP or port.
        """
        result, controller = flight_controller.FlightController.create(address)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert controller is not None

        return True, FlightInterface(cls.__create_key, controller)

    def __init__(self, class_private_create_key, controller: flight_controller.FlightController):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller

    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = self.controller.get_odometry()
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        return odometry_and_time.OdometryAndTime.create(odometry)

# pylint: enable=too-few-public-methods
