"""
Object detection in world space
"""

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class DetectionInWorld:
    """
    Typically on the ground
    """
    def __init__(self, vertices: np.ndarray, centre: np.ndarray, label: int, confidence: float):
        """
        vertices is a quadrilateral of 4 points
        centre is a point
        A point is an xy coordinate (index 0 and 1 respectively)
        """
        assert vertices.shape == (4, 2)
        assert centre.shape[0] == 2

        self.vertices = vertices
        self.centre = centre
        self.label = label
        self.confidence = confidence

# pylint: enable=too-few-public-methods
