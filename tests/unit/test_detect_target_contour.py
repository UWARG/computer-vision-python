"""
Test DetectTarget module.

"""

from typing import NamedTuple
import math
import cv2
import numpy as np
import pytest

from modules.detect_target import detect_target_contour
from modules import image_and_time
from modules import detections_and_time

BOUNDING_BOX_PRECISION_TOLERANCE = -2 / 3
CONFIDENCE_PRECISION_TOLERANCE = 2

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


class LandingPadData(NamedTuple):
    """
    Landing Pad information struct
    """

    center: tuple[int, int]
    radius: int | tuple[int, int]
    blur: bool = False
    elipse: bool = False
    angle: int = 0


easy_data = [LandingPadData(center=(1000, 400), radius=200)]

blurry_data = [
    LandingPadData(center=(1000, 500), radius=423, blur=True),
]

stretched_data = [LandingPadData(center=(1000, 500), radius=(383, 405), elipse=True)]

multiple_data = [
    LandingPadData(center=(200, 500), radius=(50, 45), blur=True, elipse=True),
    LandingPadData(center=(1590, 341), radius=250),
    LandingPadData(center=(997, 600), radius=300),
    LandingPadData(center=(401, 307), radius=(200, 150), blur=True, elipse=True),
]


def blur_img(
    bg: np.ndarray,
    center: tuple[int, int],
    radius: int = 0,
    axis_length: tuple[int, int] = (0, 0),
    angle: int = 0,
    circle_check: bool = True,
) -> np.ndarray:
    """
    Blurs an image a singular shape and adds it to the background.
    """

    bg_copy = bg.copy()
    x, y = bg_copy.shape[:2]

    mask = np.zeros((x, y), np.uint8)
    if circle_check:
        mask = cv2.circle(mask, center, radius, (215, 158, 115), -1, cv2.LINE_AA)
    else:
        mask = cv2.ellipse(mask, center, axis_length, angle, 0, 360, (215, 158, 115), -1)

    mask = cv2.blur(mask, (25, 25), 7)

    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    fg = np.zeros(bg.shape, np.uint8)
    fg[:, :, :] = [0, 0, 0]

    blended = cv2.convertScaleAbs(bg * (1 - alpha) + fg * alpha)
    return blended


def draw_circle(
    image: np.ndarray, center: tuple[int, int], radius: int, blur: bool
) -> tuple[np.ndarray, int, int]:
    """
    Draws a circle on the provided image and saves the bounding box coordinates to a text file.
    """
    x, y = center
    top_left = (max(x - radius, 0), max(y - radius, 0))
    bottom_right = (min(x + radius, image.shape[1]), min(y + radius, image.shape[0]))

    if blur:
        image = blur_img(image, center, radius=radius, circle_check=True)
        return image, top_left, bottom_right

    cv2.circle(image, center, radius, (215, 158, 115), -1)
    return image, top_left, bottom_right


def draw_ellipse(
    image: np.ndarray, center: tuple[int, int], axis_length: tuple, angle: int, blur: bool
) -> tuple[np.ndarray, int, int]:
    """
    Draws an ellipse on the provided image and saves the bounding box coordinates to a text file.
    """
    (h, k), (a, b) = center, axis_length
    rad = math.pi / 180
    ux, uy = a * math.cos(angle * rad), a * math.sin(angle * rad)  # first point on the ellipse
    vx, vy = b * math.sin(angle * rad), b * math.cos(angle * rad)
    width, height = 2 * math.sqrt(ux**2 + vx**2), 2 * math.sqrt(uy**2 + vy**2)

    top_left = (int(max(h - (0.5) * width, 0)), int(max(k - (0.5) * height, 0)))
    bottom_right = (
        int(min(h + (0.5) * width, image.shape[1])),
        int(min(k + (0.5) * height, image.shape[0])),
    )

    if blur:
        image = blur_img(image, center, axis_length=axis_length, angle=angle, circle_check=False)
        return image, top_left, bottom_right

    image = cv2.ellipse(image, center, axis_length, angle, 0, 360, (0, 0, 0), -1)
    return image, top_left, bottom_right

def create_test_case(landing_list: list[LandingPadData]) -> tuple[np.ndarray, np.ndarray]:
    """
    Genereates test cases given a data set.
    """
    image = np.full(shape=(1000, 2000, 3), fill_value=255, dtype=np.int16)

    boxes_list = []
    for landing_data in landing_list:
        if landing_data.elipse:
            axis_length, angle = landing_data.radius, landing_data.angle
            image, top_left, bottom_right = draw_ellipse(
                image, landing_data.center, axis_length, angle, landing_data.blur
            )
            boxes_list.append([1, 0] + list(top_left + bottom_right))
            continue

        image, top_left, bottom_right = draw_circle(
            image, landing_data.center, landing_data.radius, landing_data.blur
        )
        boxes_list.append([1, 0] + list(top_left + bottom_right))

    boxes_list = sorted(
        boxes_list,
        reverse=True,
        key=lambda boxes_list: abs(
            (boxes_list[4] - boxes_list[2]) * (boxes_list[5] - boxes_list[3])
        ),
    )
    boxes_list = np.array(boxes_list)
    return (image.astype(np.uint8), boxes_list)


# pylint:disable=duplicate-code
def compare_detections(
    actual: detections_and_time.DetectionsAndTime, expected: detections_and_time.DetectionsAndTime
) -> None:
    """
    Compare expected and actual detections.
    """
    assert len(actual.detections) == len(expected.detections)

    # Using integer indexing for both lists
    # pylint: disable-next=consider-using-enumerate
    for i in range(0, len(expected.detections)):
        expected_detection = expected.detections[i]
        actual_detection = actual.detections[i]

        assert expected_detection.label == actual_detection.label
        np.testing.assert_almost_equal(
            expected_detection.confidence,
            actual_detection.confidence,
            decimal=CONFIDENCE_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.x_1,
            expected_detection.x_1,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y_1,
            expected_detection.y_1,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.x_2,
            expected_detection.x_2,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y_2,
            expected_detection.y_2,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )


def create_detections(detections_from_file: np.ndarray) -> detections_and_time.DetectionsAndTime:
    """
    Create DetectionsAndTime from expected.
    Format: [confidence, label, x_1, y_1, x_2, y_2] .
    """
    assert detections_from_file.shape[1] == 6

    result, detections = detections_and_time.DetectionsAndTime.create(0)
    assert result
    assert detections is not None

    for i in range(0, detections_from_file.shape[0]):
        result, detection = detections_and_time.Detection.create(
            detections_from_file[i][2:],
            int(detections_from_file[i][1]),
            detections_from_file[i][0],
        )
        assert result
        assert detection is not None
        detections.append(detection)

    return detections


@pytest.fixture()
def detector() -> detect_target_contour.DetectTargetContour:  # type: ignore
    """
    Construct DetectTargetContour.
    """
    detection = detect_target_contour.DetectTargetContour(True)
    yield detection  # type: ignore


@pytest.fixture()
def image_easy() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load easy image.
    """

    image, _ = create_test_case(easy_data)
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def blurry_image() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load easy image.
    """

    image, _ = create_test_case(blurry_data)
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def stretched_image() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load easy image.
    """

    image, _ = create_test_case(stretched_data)
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def multiple_images() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load easy image.
    """

    image, _ = create_test_case(multiple_data)
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def expected_easy() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected an easy image detections.
    """

    _, expected = create_test_case(easy_data)
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_blur() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected an easy image detections.
    """

    _, expected = create_test_case(blurry_data)
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_stretch() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected an easy image detections.
    """

    _, expected = create_test_case(stretched_data)
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_multiple() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected an easy image detections.
    """

    _, expected = create_test_case(multiple_data)
    yield create_detections(expected)  # type: ignore


class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    def test_single_circle(
        self,
        detector: detect_target_contour.DetectTargetContour,
        image_easy: image_and_time.ImageAndTime,
        expected_easy: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(image_easy)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_easy)

    def test_blurry_circle(
        self,
        detector: detect_target_contour.DetectTargetContour,
        blurry_image: image_and_time.ImageAndTime,
        expected_blur: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(blurry_image)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_blur)

    def test_stretch(
        self,
        detector: detect_target_contour.DetectTargetContour,
        stretched_image: image_and_time.ImageAndTime,
        expected_stretch: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(stretched_image)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_stretch)

    def test_multiple(
        self,
        detector: detect_target_contour.DetectTargetContour,
        multiple_images: image_and_time.ImageAndTime,
        expected_multiple: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(multiple_images)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_multiple)
