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
    __create_key = object()

    @classmethod
    def create(cls, frame: np.ndarray) -> "tuple[bool, FrameAndTime | None]":
        """
        frame is a 2D image in RGB format
        """
        if len(frame.shape) != 3:
            return False, None

        if frame.shape[2] != 3:
            return False, None

        return True, FrameAndTime(cls.__create_key, frame)

    def __init__(self, class_private_create_key, frame: np.ndarray):
        """
        Private constructor, use create() method
        Constructor sets timestamp to current time
        """
        assert class_private_create_key is FrameAndTime.__create_key, "Use create() method"

        self.frame = frame
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
