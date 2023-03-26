"""
Combines image and timestamp together
"""

from ..common.comms.modules import generic_comms_device
from .. import message_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class ZpInput:
    """
    Combines ZP message and timestamp together
    """

    def __init__(self, port: str, baudrate: int):
        self.device = generic_comms_device.GenericCommsDevice(port, baudrate)

    def run(self):
        """
        Returns a possible FrameAndTime with current timestamp
        """
        result, message = self.device.receive()
        if not result:
            return False, None

        return True, message_and_time.MessageAndTime(message)

# pylint: enable=too-few-public-methods
