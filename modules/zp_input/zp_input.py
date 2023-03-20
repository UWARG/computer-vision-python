"""
Combines image and timestamp together
"""

from ..common.comms.modules import generic_comms_device
from .. import telemetry_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class TelemetryInput:
    """
    Combines telemetry and timestamp together
    """

    def __init__(self, port: str, baudrate: int):
        self.device = generic_comms_device.GenericCommsDevice(port, baudrate)

    def run(self):
        """
        Returns a possible FrameAndTime with current timestamp
        """
        result, telemetry = self.device.receive()
        if not result:
            return False, None

        return True, telemetry_and_time.TelemetryAndTime(telemetry)

# pylint: enable=too-few-public-methods
