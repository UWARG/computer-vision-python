"""
TODO: PointsAndTime.
"""
import copy

import cv2
import numpy as np
import pathlib
import pytest
import torch

from modules.detect_target import detect_target
from modules import image_and_time
from modules import detections_and_time


DEVICE =                        0 if torch.cuda.is_available() else "cpu"
MODEL_PATH =                    pathlib.Path("tests", "model_example", "yolov8s_ultralytics_pretrained_default.pt")
OVERRIDE_FULL =                 not torch.cuda.is_available() # CPU does not support half precision
IMAGE_BUS_PATH =                pathlib.Path("tests", "model_example", "bus.jpg")
BOUNDING_BOX_BUS_PATH =         pathlib.Path("tests", "model_example", "bounding_box_bus.txt")
IMAGE_ZIDANE_PATH =             pathlib.Path("tests", "model_example", "zidane.jpg")
BOUNDING_BOX_ZIDANE_PATH =      pathlib.Path("tests", "model_example", "bounding_box_zidane.txt")

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

@pytest.fixture()
def expected_bus():
    """
    Load expected bus bounding box.
    """
    expected_bus = np.loadtxt(BOUNDING_BOX_BUS_PATH)
    assert expected_bus is not None
    yield expected_bus

@pytest.fixture()
def expected_zidane():
    """
    Load expected Zidane bounding box.
    """
    expected_zidane = np.loadtxt(BOUNDING_BOX_ZIDANE_PATH)
    assert expected_zidane is not None
    yield expected_zidane

class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    __BOUNDING_BOX_TOLERANCE = 1e-7

    def test_single_bus_image(self,
                              detector: detect_target.DetectTarget,
                              image_bus: image_and_time.ImageAndTime,
                              expected_bus: np.ndarray):
        """
        Bus image.
        """
        # Run
        result, actual = detector.run(image_bus)

        # Test
        assert result
        assert actual is not None

        detections = actual.detections
    
        assert len(detections) == expected_bus.shape[0]

        errors = np.full((1, expected_bus.shape[1]), 1)

        for i in range(0, expected_bus.shape[0]):
            errors = np.abs(np.array([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2]) - expected_bus[i])
            assert all(e < self.__BOUNDING_BOX_TOLERANCE for e in errors)

    def test_single_zidane_image(self,
                                 detector: detect_target.DetectTarget,
                                 image_zidane: image_and_time.ImageAndTime,
                                 expected_zidane: np.ndarray):
        """
        Zidane image.
        """
        # Run
        result, actual = detector.run(image_zidane)

        # Test
        assert result
        assert actual is not None

        detections = actual.detections

        assert len(detections) == expected_zidane.shape[0]

        errors = np.full((1, expected_zidane.shape[0]), 1)

        for i in range(0, expected_zidane.shape[0]):
            errors = np.abs(np.array([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2]) - expected_zidane[i])
            assert all(e < self.__BOUNDING_BOX_TOLERANCE for e in errors)

    def test_multiple_zidane_image(self,
                                   detector: detect_target.DetectTarget,
                                   image_zidane: image_and_time.ImageAndTime,
                                   expected_zidane: np.ndarray):
        """
        Multiple Zidane images.
        """
        IMAGE_COUNT = 4

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

            assert result
            assert actual is not None
            
            detections = actual.detections

            assert len(detections) == expected_zidane.shape[0]

            errors = np.full((1, expected_zidane.shape[0]), 1)

            for i in range(0, expected_zidane.shape[0]):
                errors = np.abs(np.array([detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2]) - expected_zidane[i])
                assert all(e < self.__BOUNDING_BOX_TOLERANCE for e in errors)
