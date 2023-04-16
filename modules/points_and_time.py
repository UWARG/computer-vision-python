"""
Image points and timestamp
"""

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class PointsAndTime:
    """
    Contains image points and timestamp
    """
    def __init__(self, points: np.ndarray, timestamp: float):
        """
        Constructor asserts points are of shape (n, 2)
        """
        assert points.shape[0] > 0
        assert points.shape[1] == 2
        self.points = points
        self.timestamp = timestamp

# pylint: enable=too-few-public-methods
