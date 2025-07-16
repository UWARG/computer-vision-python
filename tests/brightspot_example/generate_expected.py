"""
Generates expected output for the brightspot detector.
"""

import pathlib

import cv2
import numpy as np

from modules import image_and_time
from modules.common.modules.logger import logger
from modules.detect_target import detect_target_brightspot


TEST_PATH = pathlib.Path("tests", "brightspot_example")

NUMBER_OF_IMAGES_DETECTIONS = 5
IMAGE_DETECTIONS_FILES = [
    pathlib.Path(f"ir_detections_{i}.png") for i in range(0, NUMBER_OF_IMAGES_DETECTIONS)
]
ANNOTATED_IMAGE_PATHS = [
    pathlib.Path(TEST_PATH, f"ir_detections_{i}_annotated.png")
    for i in range(0, NUMBER_OF_IMAGES_DETECTIONS)
]
EXPECTED_DETECTIONS_PATHS = [
    pathlib.Path(TEST_PATH, f"bounding_box_ir_detections_{i}.txt")
    for i in range(0, NUMBER_OF_IMAGES_DETECTIONS)
]

NUMBER_OF_IMAGES_NO_DETECTIONS = 2
IMAGE_NO_DETECTIONS_FILES = [
    pathlib.Path(f"ir_no_detections_{i}.png") for i in range(0, NUMBER_OF_IMAGES_NO_DETECTIONS)
]

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
    min_area_pixels=50,
    max_area_pixels=640,
    min_brightness_threshold=0,
    min_average_brightness_threshold=0,
)


def main() -> int:
    """
    Main function.
    """
    result, temp_logger = logger.Logger.create("test_logger", False)
    if not temp_logger:
        print("ERROR: Failed to create logger.")
        return 1

    detector = detect_target_brightspot.DetectTargetBrightspot(
        config=DETECT_TARGET_BRIGHTSPOT_CONFIG,
        local_logger=temp_logger,
        show_annotations=False,
        save_name="",
    )

    for image_file, annotated_image_path, expected_detections_path in zip(
        IMAGE_DETECTIONS_FILES, ANNOTATED_IMAGE_PATHS, EXPECTED_DETECTIONS_PATHS
    ):
        image_path = pathlib.Path(TEST_PATH, image_file)
        image = cv2.imread(str(image_path))  # type: ignore
        result, image_data = image_and_time.ImageAndTime.create(image)
        if not result:
            temp_logger.error(f"Failed to load image {image_path}.")
            continue

        # Get Pylance to stop complaining
        assert image_data is not None

        result, detections = detector.run(image_data)
        if not result:
            temp_logger.error(f"Unable to get detections for {image_path}.")
            continue

        # Get Pylance to stop complaining
        assert detections is not None

        detections_list = []
        image_annotated = image.copy()
        for detection in detections.detections:
            confidence = detection.confidence
            label = detection.label
            x_1 = detection.x_1
            y_1 = detection.y_1
            x_2 = detection.x_2
            y_2 = detection.y_2
            detections_list.append([confidence, label, x_1, y_1, x_2, y_2])

            cv2.rectangle(image_annotated, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 255, 0), 1)  # type: ignore

        detections_array = np.array(detections_list)

        np.savetxt(expected_detections_path, detections_array, fmt="%.6f")
        temp_logger.info(f"Expected detections saved to {expected_detections_path}.")

        result = cv2.imwrite(str(annotated_image_path), image_annotated)  # type: ignore
        if not result:
            temp_logger.error(f"Failed to write image to {annotated_image_path}.")
            continue

        temp_logger.info(f"Annotated image saved to {annotated_image_path}.")

    for image_file in IMAGE_NO_DETECTIONS_FILES:
        result, detections = detector.run(image_data)
        if result:
            temp_logger.error(f"False positive detections in {image_path}.")
            continue

        assert detections is None

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")
    else:
        print("Done!")
