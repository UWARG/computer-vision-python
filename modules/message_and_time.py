"""
ZP message and timestamp
"""
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class MessageAndTime:
    """
    Contains ZP message and timestamp
    """
    def __init__(self, message):
        """
        Constructor sets timestamp to current time
        message: 1 of TelemMessages. No type annotation due to several possible types
        """
        self.message = message
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
