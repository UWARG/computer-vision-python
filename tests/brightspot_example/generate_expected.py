"""
Generates expected output for the brightspot detector using 'ir1.png'.
"""

import pathlib

import cv2
import numpy as np

from modules.detect_target import detect_target_brightspot
from modules import image_and_time
from modules import detections_and_time


TEST_PATH = pathlib.Path("tests", "brightspot_example")

IMAGE_IR1_PATH = pathlib.Path(TEST_PATH, "ir.png")
ANNOTATED_IMAGE_PATH = pathlib.Path(TEST_PATH, "ir_annotated.png")
EXPECTED_DETECTIONS_PATH = pathlib.Path(TEST_PATH, "bounding_box_ir.txt")


def main() -> int:
    """
    Main function.
    """
    image = cv2.imread(str(IMAGE_IR1_PATH))  # type: ignore
    result, image_data = image_and_time.ImageAndTime.create(image)
    if not result or image_data is None:
        print("Failed to load image.")
        return -1

    detector = detect_target_brightspot.DetectTargetBrightspot(show_annotations=False, save_name="")

    success, detections = detector.run(image_data)
    if not success or detections is None:
        print("Detection failed or returned no detections.")
        return -1

    image_annotated = image.copy()
    for detection in detections.detections:
        x_center = int((detection.x_1 + detection.x_2) / 2)
        y_center = int((detection.y_1 + detection.y_2) / 2)
        radius_x = (detection.x_2 - detection.x_1) / 2
        radius_y = (detection.y_2 - detection.y_1) / 2
        radius = int(max(radius_x, radius_y))
        cv2.circle(image_annotated, (x_center, y_center), radius, (0, 255, 0), 1)  # type: ignore

    cv2.imwrite(str(ANNOTATED_IMAGE_PATH), image_annotated)  # type: ignore
    print(f"Annotated image saved to {ANNOTATED_IMAGE_PATH}")

    detections_list = []
    for detection in detections.detections:
        confidence = detection.confidence
        label = detection.label
        x_1 = detection.x_1
        y_1 = detection.y_1
        x_2 = detection.x_2
        y_2 = detection.y_2
        detections_list.append([confidence, label, x_1, y_1, x_2, y_2])

    detections_array = np.array(detections_list)

    np.savetxt(EXPECTED_DETECTIONS_PATH, detections_array, fmt="%.6f")
    print(f"Expected detections saved to {EXPECTED_DETECTIONS_PATH}")

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")
    else:
        print("Done!")
