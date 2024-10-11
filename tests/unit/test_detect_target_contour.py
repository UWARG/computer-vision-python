"""
Test DetectTarget module.

"""

import pathlib

import cv2
import numpy as np
import pytest
import math

from modules.detect_target import detect_target_contour
from modules import image_and_time
from modules import detections_and_time


TEST_PATH = pathlib.Path("tests", "model_example")
BG_PATH = pathlib.Path(TEST_PATH, "background.png")

BOUNDING_BOX_PRECISION_TOLERANCE = 0
CONFIDENCE_PRECISION_TOLERANCE = 2


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


class GenerateTest:
    def __init__(self, circle_data: list[list], image_file: str, txt_file: str) -> None:
        self.circle_data = circle_data[0:]
        self.image_file = image_file
        self.txt_file = txt_file
        self.image_path = "tests/model_example/"

    def save_bounding_box_annotation(self, test_case: int, boxes_list: int) -> None:
        """
        Save the bounding box annotation for the circle in the format:
        format: conf class_label x_min y_min x_max y_max
        """

        txt_file = self.image_path + self.txt_file + str(test_case) + ".txt"
        with open(txt_file, "w") as f:
            for class_label, (top_left, bottom_right) in enumerate(boxes_list):
                x_min, y_min = top_left
                x_max, y_max = bottom_right

                f.write(f"{1} {class_label} {x_min} {y_min} {x_max} {y_max}\n")
        print(f"Bounding box annotation saved to {txt_file}")

    def blur_img(
        self,
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
        fg[:, :, :] = [200, 10, 200]

        blended = cv2.convertScaleAbs(bg * (1 - alpha) + fg * alpha)
        return blended

    def draw_circle(
        self, image: np.ndarray, center: tuple[int, int], radius: int, blur: bool
    ) -> tuple[np.ndarray, int, int]:
        """
        Draws a circle on the provided image and saves the bounding box coordinates to a text file.
        """
        x, y = center
        top_left = (max(x - radius, 0), max(y - radius, 0))
        bottom_right = (min(x + radius, image.shape[1]), min(y + radius, image.shape[0]))

        if blur:
            image = self.blur_img(image, center, radius=radius, circle_check=True)
            return image, top_left, bottom_right

        cv2.circle(image, center, radius, (215, 158, 115), -1)
        return image, top_left, bottom_right

    def draw_ellipse(
        self, image: np.ndarray, center: tuple[int, int], axis_length: tuple, angle: int, blur: bool
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
            image = self.blur_img(
                image, center, axis_length=axis_length, angle=angle, circle_check=False
            )
            return image, top_left, bottom_right

        image = cv2.ellipse(image, center, axis_length, angle, 0, 360, (215, 158, 115), -1)
        return image, top_left, bottom_right

    def create_test_case(self, test_case: int) -> tuple[str, str]:
        """
        Genereates test cases given a data set.
        """
        image = cv2.imread(self.image_file)

        boxes_list = []
        for center, radius, blur, ellipse_data in self.circle_data:
            if ellipse_data[0]:
                _, axis_length, angle = ellipse_data
                image, top_left, bottom_right = self.draw_ellipse(
                    image, center, axis_length, angle, blur
                )
                boxes_list.append((top_left, bottom_right))
                continue

            image, top_left, bottom_right = self.draw_circle(image, center, radius, blur)
            boxes_list.append((top_left, bottom_right))

        self.save_bounding_box_annotation(test_case, boxes_list)

        output_image_file = f"{self.image_path}test_output_{test_case}.png"
        cv2.imwrite(output_image_file, image)
        print(f"Image with bounding box saved as {output_image_file}")
        return (output_image_file, self.image_path + self.txt_file + f"{test_case}.txt")


# ---------------------------------------------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def detector() -> detect_target_contour.DetectTargetContour:  # type: ignore
    """
    Construct DetectTargetUltralytics.
    """
    detection = detect_target_contour.DetectTargetContour()
    yield detection  # type: ignore


# ---------------------------------------------------------------------------
class TestDetector:
    """
    Tests `DetectTarget.run()` .
    """

    def test_multiple_landing_pads(
        self,
        detector: detect_target_contour.DetectTargetContour,
    ) -> None:
        """
        Multiple images.
        """

        circle_data = [
            [(200, 200), 400, False, [False, None, None]],
            [(1500, 700), 500, False, [False, None, None]],
        ]

        actual_detections, expected_detections = [], []
        circle_list = [circle_data]

        for i, circle_data in enumerate(circle_list):
            generate_test = GenerateTest(circle_data, BG_PATH, "bounding_box")
            image_file, txt_file = generate_test.create_test_case(i + 1)
            image = cv2.imread(image_file, 1)

            result, actual = image_and_time.ImageAndTime.create(image)
            assert result
            assert actual is not None

            expected = create_detections(np.loadtxt(txt_file))
            actual_detections.append(actual)
            expected_detections.append(expected)

        # Run
        outputs = []
        for i in range(0, len(circle_list)):
            output = detector.run(actual_detections[i])
            outputs.append(output)

        print(outputs)
        # Test
        for i in range(0, len(outputs)):
            output: "tuple[bool, detections_and_time.DetectionsAndTime | None]" = outputs[i]
            result, actual = output

            print(actual)
            compare_detections(actual, expected_detections[i])
