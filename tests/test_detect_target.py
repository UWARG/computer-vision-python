"""
Test DetectTarget module.
"""
import copy
import pathlib

import cv2
import numpy as np
import pytest
import torch

from modules.detect_target import detect_target
from modules import image_and_time
from modules import detections_and_time


TEST_PATH = pathlib.Path("tests", "model_example")
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL_PATH = pathlib.Path(TEST_PATH, "yolov8s_ultralytics_pretrained_default.pt")
OVERRIDE_FULL = not torch.cuda.is_available()  # CPU does not support half precision
IMAGE_BUS_PATH = pathlib.Path(TEST_PATH, "bus.jpg")
BOUNDING_BOX_BUS_PATH = pathlib.Path(TEST_PATH, "bounding_box_bus.txt")
IMAGE_ZIDANE_PATH = pathlib.Path(TEST_PATH, "zidane.jpg")
BOUNDING_BOX_ZIDANE_PATH = pathlib.Path(TEST_PATH, "bounding_box_zidane.txt")

BOUNDING_BOX_PRECISION_TOLERANCE = 0
CONFIDENCE_PRECISION_TOLERANCE = 2


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
            actual_detection.x1,
            expected_detection.x1,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y1,
            expected_detection.y1,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.x2,
            expected_detection.x2,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y2,
            expected_detection.y2,
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )


def create_detections(detections_from_file: np.ndarray) -> detections_and_time.DetectionsAndTime:
    """
    Create DetectionsAndTime from expected.
    Format: [confidence, label, x1, y1, x2, y2] .
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
def detector():
    """
    Construct DetectTarget.
    """
    detection = detect_target.DetectTarget(DEVICE, str(MODEL_PATH), OVERRIDE_FULL)
    yield detection


@pytest.fixture()
def image_bus():
    """
    Load bus image.
    """
    image = cv2.imread(str(IMAGE_BUS_PATH))  # type: ignore
    result, bus_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert bus_image is not None
    yield bus_image


@pytest.fixture()
def image_zidane():
    """
    Load Zidane image.
    """
    image = cv2.imread(str(IMAGE_ZIDANE_PATH))  # type: ignore
    result, zidane_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert zidane_image is not None
    yield zidane_image


@pytest.fixture()
def expected_bus():
    """
    Load expected bus detections.
    """
    expected = np.loadtxt(BOUNDING_BOX_BUS_PATH)
    yield create_detections(expected)


@pytest.fixture()
def expected_zidane():
    """
    Load expected Zidane detections.
    """
    expected = np.loadtxt(BOUNDING_BOX_ZIDANE_PATH)
    yield create_detections(expected)


class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    def test_single_bus_image(
        self,
        detector: detect_target.DetectTarget,
        image_bus: image_and_time.ImageAndTime,
        expected_bus: detections_and_time.DetectionsAndTime,
    ):
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(image_bus)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_bus)

    def test_single_zidane_image(
        self,
        detector: detect_target.DetectTarget,
        image_zidane: image_and_time.ImageAndTime,
        expected_zidane: detections_and_time.DetectionsAndTime,
    ):
        """
        Zidane image.
        """
        # Run
        result, actual = detector.run(image_zidane)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_zidane)

    def test_multiple_zidane_image(
        self,
        detector: detect_target.DetectTarget,
        image_zidane: image_and_time.ImageAndTime,
        expected_zidane: detections_and_time.DetectionsAndTime,
    ):
        """
        Multiple Zidane images.
        """
        image_count = 4

        input_images = []
        for _ in range(0, image_count):
            input_image = copy.deepcopy(image_zidane)
            input_images.append(input_image)

        # Run
        outputs = []
        for i in range(0, image_count):
            output = detector.run(input_images[i])
            outputs.append(output)

        # Test
        for i in range(0, image_count):
            output: "tuple[bool, detections_and_time.DetectionsAndTime | None]" = outputs[i]
            result, actual = output

            assert result
            assert actual is not None
            compare_detections(actual, expected_zidane)
