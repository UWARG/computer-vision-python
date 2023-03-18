"""
Image frame and timestamp
"""
import time

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class FrameAndTime:
    """
    Contains image frame and timestamp
    """
    def __init__(self, frame: np.ndarray):
        """
        Constructor sets timestamp to current time
        """
        self.frame = frame
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
