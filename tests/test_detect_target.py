"""
TODO: PointsAndTime.
"""
import copy

import cv2
import numpy as np
import pytest
import torch

from modules.detect_target import detect_target
from modules import image_and_time


DEVICE =                        0 if torch.cuda.is_available() else "cpu"
MODEL_PATH =                    "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"
ENABLE_HALF_PRECISION =         False
IMAGE_BUS_PATH =                "tests/model_example/bus.jpg"
IMAGE_BUS_ANNOTATED_PATH =      "tests/model_example/bus_annotated.png"
IMAGE_ZIDANE_PATH =             "tests/model_example/zidane.jpg"
IMAGE_ZIDANE_ANNOTATED_PATH =   "tests/model_example/zidane_annotated.png"


@pytest.fixture()
def detector():
    """
    Construct DetectTarget.
    """
    detection = detect_target.DetectTarget(DEVICE, MODEL_PATH, ENABLE_HALF_PRECISION)
    yield detection


@pytest.fixture()
def image_bus():
    """
    Load bus image.
    """
    image = cv2.imread(IMAGE_BUS_PATH)
    result, bus_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert bus_image is not None
    yield bus_image


@pytest.fixture()
def image_zidane():
    """
    Load Zidane image.
    """
    image = cv2.imread(IMAGE_ZIDANE_PATH)
    result, zidane_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert zidane_image is not None
    yield zidane_image


def rmse(actual: np.ndarray,
         expected: np.ndarray) -> float:
        """
        Helper function to compute root mean squared error.
        """
        mean_squared_error = np.square(actual - expected).mean()

        return np.sqrt(mean_squared_error)


def test_rmse():
        """
        Root mean squared error.
        """
        # Setup
        sample_actual = np.array([1, 2, 3, 4, 5])
        sample_expected = np.array([1.6, 2.5, 2.9, 3, 4.1])
        EXPECTED_ERROR = 0.697137

        # Run
        actual_error = rmse(sample_actual, sample_expected)

        # Test
        np.testing.assert_almost_equal(actual_error, EXPECTED_ERROR)


class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    __IMAGE_DIFFERENCE_TOLERANCE = 1

    def test_single_bus_image(self,
                              detector: detect_target.DetectTarget,
                              image_bus: image_and_time.ImageAndTime):
        """
        Bus image.
        """
        # Setup
        expected = cv2.imread(IMAGE_BUS_ANNOTATED_PATH)
        assert expected is not None

        # Run
        result, actual = detector.run(image_bus)

        # Test
        assert result
        assert actual is not None
    
        error = rmse(actual, expected)
        assert error < self.__IMAGE_DIFFERENCE_TOLERANCE

    def test_single_zidane_image(self,
                                 detector: detect_target.DetectTarget,
                                 image_zidane: image_and_time.ImageAndTime):
        """
        Zidane image.
        """
        # Setup
        expected = cv2.imread(IMAGE_ZIDANE_ANNOTATED_PATH)
        assert expected is not None

        # Run
        result, actual = detector.run(image_zidane)

        # Test
        assert result
        assert actual is not None

        error = rmse(actual, expected)
        assert error < self.__IMAGE_DIFFERENCE_TOLERANCE

    def test_multiple_zidane_image(self,
                                   detector: detect_target.DetectTarget,
                                   image_zidane: image_and_time.ImageAndTime):
        """
        Multiple Zidane images.
        """
        IMAGE_COUNT = 4

        # Setup
        expected = cv2.imread(IMAGE_ZIDANE_ANNOTATED_PATH)
        assert expected is not None

        input_images = []
        for _ in range(0, IMAGE_COUNT):
            input_image = copy.deepcopy(image_zidane)
            input_images.append(input_image)

        # Run
        outputs = []
        for i in range(0, IMAGE_COUNT):
            output = detector.run(input_images[i])
            outputs.append(output)

        # Test
        for i in range(0, IMAGE_COUNT):
            output: "tuple[bool, np.ndarray | None]" = outputs[i]
            result, actual = output
            assert result
            assert actual is not None

            error = rmse(actual, expected)
            assert error < self.__IMAGE_DIFFERENCE_TOLERANCE
