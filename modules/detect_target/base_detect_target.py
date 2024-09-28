"""
Base class for detect target.
"""

from .. import image_and_time
from .. import detections_and_time


class BaseDetectTarget:
    """
    Abstract class for detect target implementations.
    """

    def __init__(self) -> None:
        """
        Virtual method.
        """
        raise NotImplementedError

    def run(
        self, data: image_and_time.ImageAndTime
    ) -> "tuple[bool, detections_and_time.DetectionsAndTime | None]":
        """
        Virtual method.
        """
        raise NotImplementedError
