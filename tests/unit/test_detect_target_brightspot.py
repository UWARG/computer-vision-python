"""
Test DetectTargetBrightspot module.
"""

import pathlib
import copy

import cv2
import numpy as np
import pytest

from modules.detect_target import detect_target_brightspot
from modules import image_and_time
from modules import detections_and_time


TEST_PATH = pathlib.Path("tests", "brightspot_example")
IMAGE_IR1_PATH = TEST_PATH / "ir.png"
EXPECTED_DETECTIONS_PATH = pathlib.Path(TEST_PATH, "bounding_box_ir.txt")

BRIGHTSPOT_THRESHOLD = 240
BOUNDING_BOX_PRECISION_TOLERANCE = 3
CONFIDENCE_PRECISION_TOLERANCE = 6


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


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
    if detections_from_file.ndim == 1:
        detections_from_file = detections_from_file.reshape(1, -1)

    assert detections_from_file.shape[1] == 6

    result, detections = detections_and_time.DetectionsAndTime.create(0)
    assert result
    assert detections is not None

    for detection_data in detections_from_file:
        confidence, label, x_1, y_1, x_2, y_2 = detection_data
        bounds = np.array([x_1, y_1, x_2, y_2])
        result, detection = detections_and_time.Detection.create(
            bounds=bounds,
            label=int(label),
            confidence=confidence,
        )
        assert result
        assert detection is not None
        detections.append(detection)

    return detections


@pytest.fixture()
def detector() -> detect_target_brightspot.DetectTargetBrightspot:  # type: ignore
    """
    Construct DetectTargetBrightspot.
    """
    detection = detect_target_brightspot.DetectTargetBrightspot(
        show_annotations=False, save_name=""
    )
    yield detection  # type: ignore


@pytest.fixture()
def image_ir() -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load ir.png image.
    """
    image = cv2.imread(str(IMAGE_IR1_PATH))
    result, ir_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert ir_image is not None
    yield ir_image  # type: ignore


@pytest.fixture()
def expected_detections() -> detections_and_time.DetectionsAndTime:  # type: ignore
    """
    Load expected detections from file.
    """
    expected = np.loadtxt(EXPECTED_DETECTIONS_PATH)
    detections = create_detections(expected)
    yield detections  # type: ignore


class TestBrightspotDetector:
    """
    Tests `DetectTargetBrightspot.run()`.
    """

    def test_single_ir_image(
        self,
        detector: detect_target_brightspot.DetectTargetBrightspot,
        image_ir: image_and_time.ImageAndTime,
        expected_detections: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Test detection on a single IR image.
        """
        result, actual = detector.run(image_ir)
        assert result
        assert actual is not None
        compare_detections(actual, expected_detections)

    def test_multiple_ir_images(
        self,
        detector: detect_target_brightspot.DetectTargetBrightspot,
        image_ir: image_and_time.ImageAndTime,
        expected_detections: detections_and_time.DetectionsAndTime,
    ) -> None:
        """
        Test detection on multiple copies of the IR image.
        """
        image_count = 4
        input_images = [copy.deepcopy(image_ir) for _ in range(image_count)]
        outputs = [detector.run(img) for img in input_images]

        for result, actual in outputs:
            assert result
            assert actual is not None
            compare_detections(actual, expected_detections)
