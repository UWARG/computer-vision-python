"""
Creates flight controller and combines odometry data and timestamp.
"""

from modules.common.mavlink.modules import flight_controller
from modules import odometry_and_time


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
        address, TCP or UDP string.
        """ 
        result, controller = flight_controller.FlightController.create(address)

        if result == False: 
            return False, None

        return True, FlightInterface(cls.__create_key, controller)

    def __init__(self, class_private_create_key, controller):
        """
        Private constructor, use create() method.
        """
        if controller is None:
            return False, None

        assert class_private_create_key is FlightInterface.__create_key, "Use create() method"

        self.controller = controller

    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, data = self.controller.get_odometry()
        if not result:
            return False, None
        
        return odometry_and_time.OdometryAndTime.create(data)
    
# pylint: enable=too-few-public-methods
