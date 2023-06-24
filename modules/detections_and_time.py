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
    def __init__(self, bounds: np.ndarray, label: int, confidence: float):
        """
        bounds are of form x1, y1, x2, y2
        """
        assert bounds.shape[0] == 4
        # Assert every element in bounds is >= 0.0
        assert np.greater_equal(bounds, 0).all()
        assert label >= 0
        assert confidence >= 0.0
        assert confidence <= 1.0

        self.bounds = bounds
        self.label = label
        self.confidence = confidence

    def __repr__(self) -> str:
        return "cls: " + str(self.label) + ", conf: " + str(self.confidence) + ", bounds: " + repr(self.bounds)

    def get_centre(self) -> "tuple[float, float]":
        """
        Gets the xy centre of the bounding box
        """
        centre_x = self.bounds[0] + self.bounds[2]
        centre_y = self.bounds[1] + self.bounds[3]
        return centre_x, centre_y

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
        representation += "\n" + repr(self.detections)
        return representation

    def __len__(self) -> int:
        """
        Gets the number of detected objects
        """
        return len(self.detections)

    def append(self, detection: Detection):
        """
        Appends a detected object
        """
        self.detections.append(detection)

# pylint: enable=too-few-public-methods
