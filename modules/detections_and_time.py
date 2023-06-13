"""
Detection information and timestamp
"""

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class Detection:
    """
    A detected object
    """
    def __init__(self, bounds: np.ndarray, label: int, confidence: float,):
        """
        bounds are of form x1, y1, x2, y2
        """
        assert bounds.shape[0] == 4
        assert confidence >= 0.0
        assert confidence <= 1.0
        self.bounds = bounds
        self.label = label
        self.confidence = confidence

    def __repr__(self) -> str:
        return "cls: " + str(self.label) + ", conf: " + str(self.confidence) + ", bounds: " + self.bounds.__repr__()

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class DetectionsAndTime:
    """
    Contains detected object and timestamp
    """
    def __init__(self,  timestamp: float):
        self.detections = []
        self.timestamp = timestamp

    def __repr__(self) -> str:
        representation = str(self.__class__) + ", time: " + str(int(self.timestamp)) + ", size: " + str(self.size())
        representation += "\n" + self.detections.__repr__()
        return representation

    def append(self, detection: Detection):
        """
        Appends a detected object
        """
        self.detections.append(detection)

    def size(self) -> int:
        """
        Gets the number of detected objects
        """
        return len(self.detections)

# pylint: enable=too-few-public-methods
