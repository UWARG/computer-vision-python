"""
Helper functions for test_detect_target_contour.

"""

import cv2
import math
import numpy as np

LANDING_PAD_COLOR = (100, 50, 50)  # blue color

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


class BoundingBox:
    """
    Holds the data that define the generated bounding boxes.

    Attributes:
        top_left: A tuple of integers that represents top left corner of bounding box.
        bottom_right: A tuple of integers that represents bottom right corner of bounding box.
    """

    def __init__(self, top_left: tuple[int, int], bottom_right: tuple[int, int]):
        self.top_left = top_left
        self.bottom_right = bottom_right


class NumpyImage:
    """
    Holds the Numpy Array which represents an image.
    """

    def __init__(self, image: np.ndarray):
        self.image = image


class LandingPadTestData:
    """
    Struct to hold the data needed to perform the tests.

    Attributes:
        image = A numpy array that represents the image needed to be tested
        bounding_box_list: A numpy array that holds a list of expected bounding box coordinates
    """

    def __init__(self, image: NumpyImage, boxes_list: np.ndarray):
        self.image = image.image
        self.bounding_box_list = boxes_list


class LandingPadData:
    """
    Represents the data required to define and generate a landing pad.

    Attributes:
        center: The (x, y) coordinates representing the center of the landing pad.
        axis: The lengths of the semi-major and semi-minor axes of the ellipse.
        blur: Indicates whether the landing pad should have a blur effect. default: False.
        angle (int): The rotation angle of the landing pad in degrees. defaults: 0.
    """

    def __init__(
        self,
        center: tuple[int, int],
        axis: tuple[int, int],
        blur: bool = False,
        angle: int = 0,
    ):

        self.center = center
        self.axis = axis
        self.blur = blur
        self.angle = angle


def blur_image(background: np.ndarray, landing_data: LandingPadData) -> NumpyImage:
    """
    Blurs an image a singular shape, adds it to the background, and returns an image.
    """

    background_copy = background.copy()
    x, y = background_copy.shape[:2]

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
    # Brings the image back to its original color
    fg = np.full(background.shape, LANDING_PAD_COLOR, dtype=np.uint8)

    blended = (background * (1 - alpha) + fg * alpha).astype(np.uint8)
    return NumpyImage(blended)


def draw_landing_pad(
    image: np.ndarray, landing_data: LandingPadData
) -> tuple[NumpyImage, BoundingBox]:
    """
    Draws an ellipse on the provided image and saves the bounding box coordinates to a text file.
    """
    (h, k), (a, b) = landing_data.center, landing_data.axis
    rad = math.pi / 180
    ux, uy = a * math.cos(landing_data.angle * rad), a * math.sin(landing_data.angle * rad)
    vx, vy = b * math.sin(landing_data.angle * rad), b * math.cos(landing_data.angle * rad)
    width, height = 2 * math.sqrt(ux**2 + vx**2), 2 * math.sqrt(uy**2 + vy**2)

    top_left = (int(max(h - (0.5) * width, 0)), int(max(k - (0.5) * height, 0)))
    bottom_right = (
        int(min(h + (0.5) * width, image.shape[1])),
        int(min(k + (0.5) * height, image.shape[0])),
    )

    if landing_data.blur:
        image = blur_image(image, landing_data)
        return image, BoundingBox(top_left, bottom_right)

    image = cv2.ellipse(
        image,
        landing_data.center,
        landing_data.axis,
        landing_data.angle,
        0,
        360,
        LANDING_PAD_COLOR,
        -1,
    )
    return NumpyImage(image), BoundingBox(top_left, bottom_right)


def create_test(landing_list: list[LandingPadData]) -> LandingPadTestData:
    """
    Generates test cases given a data set.
    """
    image = np.full(shape=(1000, 2000, 3), fill_value=255, dtype=np.int16)

    boxes_list = []
    for landing_data in landing_list:
        np_image, bounding_box = draw_landing_pad(image, landing_data)
        image = np_image.image

        boxes_list.append([1, 0] + list(bounding_box.top_left + bounding_box.bottom_right))

    boxes_list = sorted(
        boxes_list,
        reverse=True,
        key=lambda box: abs((box[4] - box[2]) * (box[5] - box[3])),
    )

    boxes_list = np.array(boxes_list)
    image = image.astype(np.uint8)
    return LandingPadTestData(NumpyImage(image), boxes_list)
