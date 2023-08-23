"""
Detection information and timestamp.
"""

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class Detection:
    """
    A detected object in image space.
    """
    __create_key = object()

    @classmethod
    def create(cls,
               bounds: np.ndarray,
               label: int,
               confidence: float) -> "tuple[bool, Detection | None]":
        """
        bounds are of form x1, y1, x2, y2 .
        """
        # Check every element in bounds is >= 0.0
        if bounds.shape != (4,) or not np.greater_equal(bounds, 0.0).all():
            return False, None

        # n1 <= n2
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            return False, None

        if label < 0:
            return False, None

        if confidence < 0.0 or confidence > 1.0:
            return False, None

        return True, Detection(cls.__create_key, bounds, label, confidence)

    def __init__(self, class_private_create_key, bounds: np.ndarray, label: int, confidence: float):
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Detection.__create_key, "Use create() method"

        # Mixing letters and numbers confuses Pylint
        # pylint: disable=invalid-name
        self.x1 = bounds[0]
        self.y1 = bounds[1]
        self.x2 = bounds[2]
        self.y2 = bounds[3]
        # pylint: enable=invalid-name
        self.label = label
        self.confidence = confidence

    def __repr__(self) -> str:
        representation = \
            "cls: " + str(self.label) \
            + ", conf: " + str(self.confidence) \
            + ", bounds: " \
                + str(self.x1) + " " \
                + str(self.y1) + " " \
                + str(self.x2) + " " \
                + str(self.y2)

        return representation

    def get_centre(self) -> "tuple[float, float]":
        """
        Gets the xy centre of the bounding box.
        """
        centre_x = (self.x1 + self.x2) / 2
        centre_y = (self.y1 + self.y2) / 2
        return centre_x, centre_y

    def get_corners(self) -> "list[tuple[float, float]]":
        """
        Gets the xy corners of the bounding box.
        """
        top_left = self.x1, self.y1
        top_right = self.x2, self.y1
        bottom_left = self.x1, self.y2
        bottom_right = self.x2, self.y2
        return [top_left, top_right, bottom_left, bottom_right]

# pylint: enable=too-few-public-methods


# Basically a struct
# pylint: disable=too-few-public-methods
class DetectionsAndTime:
    """
    Contains detected object and timestamp.
    """
    def __init__(self,  timestamp: float):
        self.detections = []
        self.timestamp = timestamp

    def __repr__(self) -> str:
        representation = \
            str(self.__class__) \
            + ", time: " + str(int(self.timestamp)) \
            + ", size: " + str(len(self))

        representation += "\n" + repr(self.detections)
        return representation

    def __len__(self) -> int:
        """
        Gets the number of detected objects.
        """
        return len(self.detections)

    def append(self, detection: Detection):
        """
        Appends a detected object.
        """
        self.detections.append(detection)

# pylint: enable=too-few-public-methods
