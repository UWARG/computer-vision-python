"""
Test contour detection module.
"""

import numpy as np
import pytest

from modules import detections_and_time
from modules.detect_target import detect_target_contour
from modules.common.modules.logger import logger
from tests.unit import generate_detect_target_contour

BOUNDING_BOX_PRECISION_TOLERANCE = -1  # Tolerance > 1
CONFIDENCE_PRECISION_TOLERANCE = 2
LOGGER_NAME = ""


# Test functions use test fixture signature names and access class privates
# pylint: disable=protected-access,redefined-outer-name, duplicate-code

@pytest.fixture
def single_circle() -> generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes:
    """
    Loads the data for the single basic circle.
    """
    options = [
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(300, 400), axis=(200, 200), blur=False, angle=0
        )
    ]

    test_data = generate_detect_target_contour.create_test(options)
    yield test_data


@pytest.fixture
def blurry_circle() -> generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes:
    """
    Loads the data for the single blury circle.
    """
    options = [
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(1000, 500), axis=(423, 423), blur=True, angle=0
        ),
    ]

    test_data = generate_detect_target_contour.create_test(options)
    yield test_data


@pytest.fixture
def stretched_circle() -> generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes:
    """
    Loads the data for the single stretched circle.
    """
    options = [
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(1000, 500), axis=(383, 405), blur=False, angle=0
        )
    ]

    test_data = generate_detect_target_contour.create_test(options)
    yield test_data


@pytest.fixture
def multiple_circles() -> generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes:
    """
    Loads the data for the multiple stretched circles.
    """
    options = [
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(997, 600), axis=(300, 300), blur=False, angle=0
        ),
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(1590, 341), axis=(250, 250), blur=False, angle=0
        ),
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(200, 500), axis=(50, 45), blur=True, angle=0
        ),
        generate_detect_target_contour.LandingPadImageConfig(
            centre=(401, 307), axis=(200, 150), blur=True, angle=0
        ),
    ]

    test_data = generate_detect_target_contour.create_test(options)
    yield test_data


@pytest.fixture()
def detector() -> detect_target_contour.DetectTargetContour:  # type: ignore
    """
    Construct DetectTargetContour.
    """
    result, test_logger = logger.Logger.create(LOGGER_NAME, False)

    assert result
    assert test_logger is not None

    detection = detect_target_contour.DetectTargetContour(test_logger, False)
    yield detection  # type: ignore


def compare_detections(
    actual_and_expected_detections: generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes
) -> None:
    """
    Compare expected and actual detections.
    
    actual_and_expected_detections: Test data containing both actual image and time and expected bounding boxes.
    """
    actual = actual_and_expected_detections.image_and_time_data.detector  
    expected = actual_and_expected_detections.bounding_box_list

    assert len(actual) == len(expected)

    # Ordered for the mapping to the corresponding detections
    sorted_actual_detections = sorted(
        actual,
        reverse=True,
        key=lambda box: abs((box.x_2 - box.x_1) * (box.y_2 - box.y_1)),
    )

    for i, expected_detection in enumerate(expected):
        actual_detection = sorted_actual_detections[i]

        # Check label and confidence 
        assert actual_detection.label == expected_detection[1]
        np.testing.assert_almost_equal(
            actual_detection.confidence,
            expected_detection[0],
            decimal=CONFIDENCE_PRECISION_TOLERANCE,
        )

        # Check bounding box coordinates
        np.testing.assert_almost_equal(
            actual_detection.x_1,
            expected_detection[2],
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y_1,
            expected_detection[3],
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.x_2,
            expected_detection[4],
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )

        np.testing.assert_almost_equal(
            actual_detection.y_2,
            expected_detection[5],
            decimal=BOUNDING_BOX_PRECISION_TOLERANCE,
        )


class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    def test_single_circle(
        self,
        detector: detect_target_contour.DetectTargetContour,
        single_circle: generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes,
    ) -> None:
        """
        Run the detection for the single circular landing pad.
        """
        # Run
        result, actual = detector.run(single_circle.image_and_time_data)

        # Test
        assert result
        assert actual is not None

        # Create new object with actual detections
        test_data = generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes(
            actual,
            single_circle.bounding_box_list
        )
        compare_detections(test_data)

    def test_blurry_circle(
        self,
        detector: detect_target_contour.DetectTargetContour,
        blurry_circle: generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes,
    ) -> None:
        """
        Run the detection for the blury cicular circle.
        """
        # Run
        result, actual = detector.run(blurry_circle.image_and_time_data)

        # Test
        assert result
        assert actual is not None

        compare_detections(blurry_circle)

    def test_stretched_circle(
        self,
        detector: detect_target_contour.DetectTargetContour,
        stretched_circle: generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes,
    ) -> None:
        """
        Run the detection for a single stretched circular landing pad.
        """
        # Run
        result, actual = detector.run(stretched_circle.image_and_time_data)

        # Test
        assert result
        assert actual is not None

        compare_detections(stretched_circle)

    def test_multiple_circles(
        self,
        detector: detect_target_contour.DetectTargetContour,
        multiple_circles: generate_detect_target_contour.InputImageAndTimeAndExpectedBoundingBoxes,
    ) -> None:
        """
        Run the detection for the multiple landing pads.
        """
        # Run
        result, actual = detector.run(multiple_circles.image_and_time_data)

        # Test
        assert result
        assert actual is not None

        compare_detections(multiple_circles.bounding_box_list)
