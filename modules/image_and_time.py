"""
Image and timestamp.
"""
import time

import numpy as np


# Basically a struct
# pylint: disable=too-few-public-methods
class ImageAndTime:
    """
    Contains image and timestamp.
    """
    __create_key = object()

    @classmethod
    def create(cls, image: np.ndarray) -> "tuple[bool, ImageAndTime | None]":
        """
        image: 2D image in RGB format.
        """
        if len(image.shape) != 3:
            return False, None

        if image.shape[2] != 3:
            return False, None

        return True, ImageAndTime(cls.__create_key, image)

    def __init__(self, class_private_create_key, image: np.ndarray):
        """
        Private constructor, use create() method.
        Constructor sets timestamp to current time.
        """
        assert class_private_create_key is ImageAndTime.__create_key, "Use create() method"

        self.image = image
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
