"""
Helper functions for `test_detect_target_contour.py`.

"""

import cv2
import math
import numpy as np


LANDING_PAD_COLOUR_BLUE = (100, 50, 50)  # BGR


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


class LandingPadImageConfig:
    """
    Represents the data required to define and generate a landing pad.
    """
    def __init__(
        self,
        center: tuple[int, int],
        axis: tuple[int, int],
        blur: bool,
        angle: float,
    ):
        """

        center: The (x, y) coordinates representing the center of the landing pad.
        axis: The pixel lengths of the semi-major and semi-minor axes of the ellipse.
        blur: Indicates whether the landing pad should have a blur effect.
        angle: The rotation angle of the landing pad in degrees clockwise, where 0.0 degrees
            is where both semi major and minor are aligned with the x and y-axis respectively (0.0 <= angle <= 360.0).
        """
        self.center = center
        self.axis = axis
        self.blur = blur
        self.angle = angle


class NumpyImage:
    """
    Holds the Numpy Array which represents an image.
    """
    def __init__(self, image: np.ndarray):
        """
        image: A numpy array that represents the image.
        """
        self.image = image


class BoundingBox:
    """
    Holds the data that define the generated bounding boxes.
    """
    def __init__(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        """
        top_left: x, y coordinates representing the top left corner of the bounding box on an image.
        bottom_right: x, y coordinates representing the bottom right corner of the bounding box on an image.
        """
        self.top_left = top_left
        self.bottom_right = bottom_right


class InputImageAndExpectedBoundingBoxes:
    '''
    Struct to hold the data needed to perform the tests.
    '''
    def __init__(self, image: np.ndarray, boxes_list: np.ndarray):
        """
        image = A numpy array that represents the image needed to be tested.
        bounding_box_list: A numpy array that holds a list of expected bounding box coordinates.
        """
        self.image = image
        self.bounding_box_list = boxes_list


def add_blurred_landing_pad(
    background: np.ndarray, landing_data: LandingPadImageConfig
) -> NumpyImage:
    """
    Blurs an image and adds a singular lading pad to the background.

    background: A numpy image.
    landing_data: Landing pad data for the landing pad to be blurred and added.


    Returns: Image with the landing pad.
    """
    x, y = background.shape[:2]

    mask = np.zeros((x, y), np.uint8)
    mask = cv2.ellipse(
        mask,
        landing_data.center,
        landing_data.axis,
        landing_data.angle,
        0,
        360,
        255,
        -1,
    )

    mask = cv2.blur(mask, (25, 25), 7)

    alpha = mask[:, :, np.newaxis] / 255.0
    # Brings the image back to its original colour.
    fg = np.full(background.shape, LANDING_PAD_COLOUR_BLUE, dtype=np.uint8)

    blended = (background * (1 - alpha) + fg * alpha).astype(np.uint8)
    return NumpyImage(blended)


def draw_landing_pad(
    image: np.ndarray, landing_data: LandingPadImageConfig
) -> tuple[NumpyImage, BoundingBox]:
    """
    Draws a single landing pad on the provided image and saves the bounding box coordinates to a text file.

    image: The image to add a landing pad to.
    landing_data: Landing pad data for the landing pad to be added.

    Returns: Image with landing pad and the bounding box for the drawn landing pad.
    """
    (h, k), (a, b) = landing_data.center, landing_data.axis
    angle_in_rad = math.radians(landing_data.angle)
    ux, uy = a * math.cos(angle_in_rad), a * math.sin(angle_in_rad)
    vx, vy = b * math.sin(angle_in_rad), b * math.cos(angle_in_rad)
    width, height = 2 * math.sqrt(ux**2 + vx**2), 2 * math.sqrt(uy**2 + vy**2)

    top_left = (int(max(h - (0.5) * width, 0)), int(max(k - (0.5) * height, 0)))
    bottom_right = (
        int(min(h + (0.5) * width, image.shape[1])),
        int(min(k + (0.5) * height, image.shape[0])),
    )

    bounding_box = BoundingBox(top_left, bottom_right)

    if landing_data.blur:
        image = add_blurred_landing_pad(image, landing_data)
        return image, bounding_box

    image = cv2.ellipse(
        image,
        landing_data.center,
        landing_data.axis,
        landing_data.angle,
        0,
        360,
        LANDING_PAD_COLOUR_BLUE,
        -1,
    )
    return NumpyImage(image), bounding_box


def create_test(landing_list: list[LandingPadImageConfig]) -> InputImageAndExpectedBoundingBoxes:
    """
    Generates test cases given a data set.

    landing_list: List of landing pad data to be generated.

    Returns: The image and expected bounding box.
    """
    image = np.full(
        shape=(1000, 2000, 3), fill_value=255, dtype=np.int16
    )
    confidence_and_label = [1, 0]

    # List to hold the bounding boxes.
    boxes_list = []

    for landing_data in landing_list:
        image_wrapper, bounding_box = draw_landing_pad(image, landing_data)
        image = image_wrapper.image
        boxes_list.append(
            confidence_and_label + list(bounding_box.top_left + bounding_box.bottom_right)
        )

    # Calculates the area of the bounding box.
    boxes_list = sorted(
        boxes_list,
        reverse=True,
        key=lambda box: abs((box[4] - box[2]) * (box[5] - box[3])),
    )

    boxes_list = np.array(boxes_list)
    image = image.astype(np.uint8)

    return InputImageAndExpectedBoundingBoxes(image, boxes_list)
