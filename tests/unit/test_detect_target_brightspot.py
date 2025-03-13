"""
Test DetectTargetBrightspot module.
"""

import pathlib

import cv2
import numpy as np
import pytest

from modules import detections_and_time
from modules import image_and_time
from modules.common.modules.logger import logger
from modules.detect_target import detect_target_brightspot


TEST_PATH = pathlib.Path("tests", "brightspot_example")

NUMBER_OF_IMAGES_DETECTIONS = 5
IMAGE_DETECTIONS_FILES = [
    pathlib.Path(f"ir_detections_{i}.png") for i in range(0, NUMBER_OF_IMAGES_DETECTIONS)
]
EXPECTED_DETECTIONS_PATHS = [
    pathlib.Path(TEST_PATH, f"bounding_box_ir_detections_{i}.txt")
    for i in range(0, NUMBER_OF_IMAGES_DETECTIONS)
]
DETECTION_TEST_CASES = list(zip(IMAGE_DETECTIONS_FILES, EXPECTED_DETECTIONS_PATHS))

NUMBER_OF_IMAGES_NO_DETECTIONS = 2
IMAGE_NO_DETECTIONS_FILES = [
    pathlib.Path(f"ir_no_detections_{i}.png") for i in range(0, NUMBER_OF_IMAGES_NO_DETECTIONS)
]
NO_DETECTION_TEST_CASES = IMAGE_NO_DETECTIONS_FILES

BOUNDING_BOX_PRECISION_TOLERANCE = 3
CONFIDENCE_PRECISION_TOLERANCE = 6

# Config is identical to test_detect_target_worker.py
# pylint: disable=duplicate-code
DETECT_TARGET_BRIGHTSPOT_CONFIG = detect_target_brightspot.DetectTargetBrightspotConfig(
    brightspot_percentile_threshold=99.9,
    filter_by_color=True,
    blob_color=255,
    filter_by_circularity=False,
    min_circularity=0.01,
    max_circularity=1,
    filter_by_inertia=True,
    min_inertia_ratio=0.2,
    max_inertia_ratio=1,
    filter_by_convexity=False,
    min_convexity=0.01,
    max_convexity=1,
    filter_by_area=True,
    min_area_pixels=30,
    max_area_pixels=640,
    min_brightness_threshold=50,
)
# pylint: enable=duplicate-code

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name,duplicate-code


def compare_detections(
    actual: detections_and_time.DetectionsAndTime, expected: detections_and_time.DetectionsAndTime
) -> None:
    """
    Compare expected and actual detections.
    """
    assert len(actual.detections) == len(expected.detections)

    for actual_detection, expected_detection in zip(actual.detections, expected.detections):
        assert expected_detection.label == actual_detection.label

        np.testing.assert_almost_equal(
            actual_detection.confidence,
            expected_detection.confidence,
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
    Create DetectionsAndTime from expected detections.
    Format: [confidence, label, x_1, y_1, x_2, y_2].
    """
    result, detections = detections_and_time.DetectionsAndTime.create(0)
    assert result
    assert detections is not None

    if detections_from_file.size == 0:
        return detections

    if detections_from_file.ndim == 1:
        detections_from_file = detections_from_file.reshape(1, -1)

    assert detections_from_file.shape[1] == 6

    for detection_data in detections_from_file:
        confidence, label, x_1, y_1, x_2, y_2 = detection_data
        bounds = np.array([x_1, y_1, x_2, y_2])
        result, detection = detections_and_time.Detection.create(bounds, int(label), confidence)
        assert result
        assert detection is not None
        detections.append(detection)

    return detections


@pytest.fixture()
def detector() -> detect_target_brightspot.DetectTargetBrightspot:  # type: ignore
    """
    Construct DetectTargetBrightspot.
    """
    result, test_logger = logger.Logger.create("test_logger", False)

    assert result
    assert test_logger is not None

    detection = detect_target_brightspot.DetectTargetBrightspot(
        DETECT_TARGET_BRIGHTSPOT_CONFIG, test_logger
    )
    yield detection  # type: ignore


@pytest.fixture(params=DETECTION_TEST_CASES)
def image_ir_detections(request: pytest.FixtureRequest) -> tuple[image_and_time.ImageAndTime, detections_and_time.DetectionsAndTime]:  # type: ignore
    """
    Load image and its corresponding expected detections.
    """
    image_file, expected_detections_file = request.param

    image_path = pathlib.Path(TEST_PATH, image_file)
    image = cv2.imread(str(image_path))
    assert image is not None

    result, ir_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert ir_image is not None

    expected = np.loadtxt(expected_detections_file)
    detections = create_detections(expected)

    yield ir_image, detections  # type: ignore


@pytest.fixture(params=NO_DETECTION_TEST_CASES)
def image_ir_no_detections(request: pytest.FixtureRequest) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load image with no detections.
    """
    image_file = request.param

    image_path = pathlib.Path(TEST_PATH, image_file)
    image = cv2.imread(str(image_path))
    assert image is not None

    result, ir_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert ir_image is not None

    yield ir_image  # type: ignore


class TestBrightspotDetector:
    """
    Tests `DetectTargetBrightspot.run()`.
    """

    def test_images_with_detections(
        self,
        detector: detect_target_brightspot.DetectTargetBrightspot,
        image_ir_detections: tuple[
            image_and_time.ImageAndTime, detections_and_time.DetectionsAndTime
        ],
    ) -> None:
        """
        Test detection on images where detections are expected.
        """
        image, expected_detections = image_ir_detections

        result, actual = detector.run(image)

        assert result
        assert actual is not None

        compare_detections(actual, expected_detections)

    def test_images_no_detections(
        self,
        detector: detect_target_brightspot.DetectTargetBrightspot,
        image_ir_no_detections: image_and_time.ImageAndTime,
    ) -> None:
        """
        Test detection on images where no detections are expected.
        """
        image = image_ir_no_detections

        result, actual = detector.run(image)

        assert result is False
        assert actual is None
