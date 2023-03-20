"""
Telemetry and timestamp
"""
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class TelemetryAndTime:
    """
    Contains telemetry and timestamp
    """
    def __init__(self, telemetry):
        """
        Constructor sets timestamp to current time
        telemetry: 1 of TelemMessages
        """
        self.telemetry = telemetry
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
