"""
Test Contour Detection module.

"""

import numpy as np
import pytest

from tests.unit.generate_detect_target_contour import (
    LandingPadImageConfig,
    InputImageAndExpectedBoundingBoxes,
    create_test,
)
from modules import detections_and_time
from modules import image_and_time
from modules.detect_target import detect_target_contour
from modules.common.modules.logger import logger  # Changed from relative to absolute import


BOUNDING_BOX_PRECISION_TOLERANCE = -2 / 3  # Tolerance > 1
CONFIDENCE_PRECISION_TOLERANCE = 2
LOGGER_NAME = ""

_, test_logger = logger.Logger.create(LOGGER_NAME, False)

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture
def single_circle() -> InputImageAndExpectedBoundingBoxes:  # type: ignore
    """
    Loads the data for the single basic circle.
    """
    options = [LandingPadImageConfig(center=(300, 400), axis=(200, 200), blur=False, angle=0)]

    test_data = create_test(options)
    yield test_data


@pytest.fixture
def single_blurry_circle() -> InputImageAndExpectedBoundingBoxes:  # type: ignore
    """
    Loads the data for the single blury circle.
    """
    options = [
        LandingPadImageConfig(center=(1000, 500), axis=(423, 423), blur=True, angle=0),
    ]

    test_data = create_test(options)
    yield test_data


@pytest.fixture
def single_stretched_circle() -> InputImageAndExpectedBoundingBoxes:  # type: ignore
    """
    Loads the data for the single stretched circle.
    """
    options = [LandingPadImageConfig(center=(1000, 500), axis=(383, 405), blur=False, angle=0)]

    test_data = create_test(options)
    yield test_data


@pytest.fixture
def multiple_circles() -> InputImageAndExpectedBoundingBoxes:  # type: ignore
    """
    Loads the data for the multiple stretched circles.
    """
    options = [
        LandingPadImageConfig(center=(997, 600), axis=(300, 300), blur=False, angle=0),
        LandingPadImageConfig(center=(1590, 341), axis=(250, 250), blur=False, angle=0),
        LandingPadImageConfig(center=(200, 500), axis=(50, 45), blur=True, angle=0),
        LandingPadImageConfig(center=(401, 307), axis=(200, 150), blur=True, angle=0),
    ]

    test_data = create_test(options)
    yield test_data


@pytest.fixture()
def detector() -> detect_target_contour.DetectTargetContour:  # type: ignore
    """
    Construct DetectTargetContour.
    """
    detection = detect_target_contour.DetectTargetContour(test_logger, False)
    yield detection  # type: ignore


@pytest.fixture()
def image_easy(single_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load the single basic landing pad.
    """

    image = single_circle.image
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def blurry_image(single_blurry_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load the single blurry landing pad.
    """

    image = single_blurry_circle.image
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def stretched_image(single_stretched_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load the single stretched landing pad.
    """

    image = single_stretched_circle.image
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def multiple_images(multiple_circles: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load the multiple landing pads.
    """

    image = multiple_circles.image
    result, actual_image = image_and_time.ImageAndTime.create(image)
    assert result
    assert actual_image is not None
    yield actual_image  # type: ignore


@pytest.fixture()
def expected_easy(single_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected a basic image detections.
    """

    expected = single_circle.bounding_box_list
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_blur(single_blurry_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected the blured pad image detections.
    """

    expected = single_blurry_circle.bounding_box_list
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_stretch(single_stretched_circle: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected a stretched pad image detections.
    """

    expected = single_stretched_circle.bounding_box_list
    yield create_detections(expected)  # type: ignore


@pytest.fixture()
def expected_multiple(multiple_circles: InputImageAndExpectedBoundingBoxes) -> image_and_time.ImageAndTime:  # type: ignore
    """
    Load expected multiple pads image detections.
    """

    expected = multiple_circles.bounding_box_list
    yield create_detections(expected)  # type: ignore


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
        Run the detection for the single landing pad.
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
        Run the detection for the blury circle.
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
        Run the detection for the single stretched landing pad.
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
        Run the detection for the multiple landing pads.
        """
        # Run
        result, actual = detector.run(multiple_images)

        # Test
        assert result
        assert actual is not None

        compare_detections(actual, expected_multiple)
