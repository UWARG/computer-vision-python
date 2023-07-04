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
    __create_key = object()

    @classmethod
    def create(cls, bounds: np.ndarray, label: int, confidence: float) -> "tuple[bool, Detection | None]":
        """
        bounds are of form x1, y1, x2, y2
        """
        # Check every element in bounds is >= 0.0
        if bounds.shape[0] != 4 or not np.greater_equal(bounds, 0.0).all():
            return False, None

        if label < 0:
            return False, None

        if confidence < 0.0 or confidence > 1.0:
            return False, None

        return True, Detection(cls.__create_key, bounds, label, confidence)

    def __init__(self, class_private_create_key, bounds: np.ndarray, label: int, confidence: float):
        """
        Private constructor, use create() method
        """
        assert class_private_create_key is Detection.__create_key, "Use create() method"

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
        representation = str(self.__class__) + ", time: " + str(int(self.timestamp)) + ", size: " + str(len(self))
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
