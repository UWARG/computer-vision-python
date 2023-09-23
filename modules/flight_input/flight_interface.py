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
        if address == "":
            return False, None
        
        assert cls.__create_key is FlightInterface.__create_key, "Use create() method"
        
        return True, FlightInterface(address)

    def __init__(self, address: str):
        """
        Private constructor, use create() method.
        """
        self.result, self.controller = flight_controller.FlightController.create(address)
    
    def run(self) -> "tuple[bool, odometry_and_time.OdometryAndTime]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, data = self.controller.get_odometry()
        if not result:
            return False, None
        
        return odometry_and_time.OdometryAndTime.create(data)
    
# pylint: enable=too-few-public-methods
