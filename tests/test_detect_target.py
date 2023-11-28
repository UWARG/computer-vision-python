"""
TODO: PointsAndTime.
"""
import copy

import cv2
import numpy as np
import pytest
import torch
import ultralytics

from modules.detect_target import detect_target
from modules import image_and_time
from modules import detections_and_time


DEVICE =                        0 if torch.cuda.is_available() else "cpu"
MODEL_PATH =                    "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"
OVERRIDE_FULL =                 False  # Tests are able to handle both full and half precision.
IMAGE_BUS_PATH =                "tests/model_example/bus.jpg"
IMAGE_BUS_ANNOTATED_PATH =      "tests/model_example/bus_annotated.png"
IMAGE_ZIDANE_PATH =             "tests/model_example/zidane.jpg"
IMAGE_ZIDANE_ANNOTATED_PATH =   "tests/model_example/zidane_annotated.png"

model = ultralytics.YOLO(MODEL_PATH)

@pytest.fixture()
def detector():
    """
    Construct DetectTarget.
    """
    detection = detect_target.DetectTarget(DEVICE, MODEL_PATH, OVERRIDE_FULL)
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
        EXPECTED_ERROR = np.sqrt(0.486)

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
        image = cv2.imread(IMAGE_BUS_PATH)
        prediction = model.predict(
            source=image,
            half=True,
            stream=False,
        )

        boxes = prediction[0].boxes
        expected = boxes.xyxy.detach().cpu().numpy()
        assert expected is not None

        # Run
        result, actual = detector.run(image_bus)
        detections = actual.detections

        # Test
        assert result
        assert actual is not None
        assert detections is not None
    
        error = 0

        for i in range(0, len(detections)):
            error += rmse([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2], expected[i])
        assert (error / len(detections)) < self.__IMAGE_DIFFERENCE_TOLERANCE

    def test_single_zidane_image(self,
                                 detector: detect_target.DetectTarget,
                                 image_zidane: image_and_time.ImageAndTime):
        """
        Zidane image.
        """
        # Setup
        image = cv2.imread(IMAGE_ZIDANE_PATH)
        prediction = model.predict(
            source=image,
            half=True,
            stream=False,
        )

        boxes = prediction[0].boxes
        expected = boxes.xyxy.detach().cpu().numpy()
        assert expected is not None

        # Run
        result, actual = detector.run(image_zidane)
        detections = actual.detections

        # Test
        assert result
        assert actual is not None
        assert detections is not None

        error = 0

        for i in range(0, len(detections)):
            error += rmse([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2], expected[i])
        assert (error / len(detections)) < self.__IMAGE_DIFFERENCE_TOLERANCE

    def test_multiple_zidane_image(self,
                                   detector: detect_target.DetectTarget,
                                   image_zidane: image_and_time.ImageAndTime):
        """
        Multiple Zidane images.
        """
        IMAGE_COUNT = 4

        # Setup
        image = cv2.imread(IMAGE_ZIDANE_PATH)
        prediction = model.predict(
            source=image,
            half=True,
            stream=False,
        )

        boxes = prediction[0].boxes
        expected = boxes.xyxy.detach().cpu().numpy()
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
            output: "tuple[bool, detections_and_time.DetectionsAndTime | None]" = outputs[i]
            result, actual = output
            
            detections = actual.detections

            assert result
            assert actual is not None
            assert detections is not None

            error = 0

            for i in range(0, len(detections)):
                error += rmse([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2], expected[i])
            assert (error / len(detections)) < self.__IMAGE_DIFFERENCE_TOLERANCE
