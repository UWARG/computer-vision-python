"""
Detection information and timestamp.
"""

import numpy as np


class Detection:
    """
    A detected object in image space.
    """

    __create_key = object()

    @classmethod
    def create(
        cls, bounds: np.ndarray, label: int, confidence: float
    ) -> "tuple[bool, Detection | None]":
        """
        bounds are of form x_1, y_1, x_2, y_2 .
        """
        # Check every element in bounds is >= 0.0
        if bounds.shape != (4,) or not np.greater_equal(bounds, 0.0).all():
            return False, None

        # n_1 <= n_2
        if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            return False, None

        if label < 0:
            return False, None

        if confidence < 0.0 or confidence > 1.0:
            return False, None

        return True, Detection(cls.__create_key, bounds, label, confidence)

    def __init__(
        self, class_private_create_key: object, bounds: np.ndarray, label: int, confidence: float
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is Detection.__create_key, "Use create() method"

        self.x_1 = bounds[0]
        self.y_1 = bounds[1]
        self.x_2 = bounds[2]
        self.y_2 = bounds[3]

        self.label = label
        self.confidence = confidence

    def __str__(self) -> str:
        """
        To string.
        """
        representation = f"cls: {self.label}, conf: {self.confidence}, bounds: {self.x_1} {self.y_1} {self.x_2} {self.y_2}"
        return representation

    def get_centre(self) -> "tuple[float, float]":
        """
        Gets the xy centre of the bounding box.
        """
        centre_x = (self.x_1 + self.x_2) / 2
        centre_y = (self.y_1 + self.y_2) / 2
        return centre_x, centre_y

    def get_corners(self) -> "list[tuple[float, float]]":
        """
        Gets the xy corners of the bounding box.
        """
        top_left = self.x_1, self.y_1
        top_right = self.x_2, self.y_1
        bottom_left = self.x_1, self.y_2
        bottom_right = self.x_2, self.y_2
        return [top_left, top_right, bottom_left, bottom_right]


class DetectionsAndTime:
    """
    Contains detected object and timestamp.
    """

    __create_key = object()

    @classmethod
    def create(cls, timestamp: float) -> "tuple[bool, DetectionsAndTime | None]":
        """
        Sets timestamp to current time.
        """
        # Check if timestamp is positive
        if timestamp < 0.0:
            return False, None

        return True, DetectionsAndTime(cls.__create_key, timestamp)

    def __init__(self, class_private_create_key: object, timestamp: float) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is DetectionsAndTime.__create_key, "Use create() method"

        self.detections = []
        self.timestamp = timestamp

    def __str__(self) -> str:
        """
        To string.
        """
        representation = f"""{self.__class__}, time: {int(self.timestamp)}, size: {len(self)}
{self.detections}"""

        return representation

    def __len__(self) -> int:
        """
        Gets the number of detected objects.
        """
        return len(self.detections)

    def append(self, detection: Detection) -> None:
        """
        Appends a detected object.
        """
        self.detections.append(detection)
