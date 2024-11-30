"""
Run bright spot detection on multiple images.
"""

import pathlib
import cv2
from modules.detect_target import detect_target_brightspot
from modules import image_and_time
from modules.common.modules.logger import logger

# Paths
IMAGE_DIR = pathlib.Path("tests/brightspot_example")
IMAGE_FILES = [
    "ir1.png",
    "ir2.png",
    "ir3.png",
    "ir4.png",
    "ir5.png",
    "ir6.png",
    "ir7.png",
]  # List of image files
SAVE_DIR = pathlib.Path("output")


def main() -> None:
    """
    Run bright spot detection on multiple images.
    """
    # Initialize logger
    result, local_logger = logger.Logger.create("brightspot_logger", False)
    if not result or local_logger is None:
        print("Failed to create logger.")
        return

    # Ensure the save directory exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for image_file in IMAGE_FILES:
        image_path = IMAGE_DIR / image_file

        # Initialize the detector
        detector = detect_target_brightspot.DetectTargetBrightspot(
            local_logger=local_logger,
            show_annotations=True,
            save_name=image_file,
        )

        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image from {image_path}")
            continue

        # Create ImageAndTime object
        result, image_data = image_and_time.ImageAndTime.create(image)
        if not result or image_data is None:
            print(f"Failed to create ImageAndTime object for {image_path}")
            continue

        # Run the detector
        success, detections = detector.run(image_data)
        if not success or detections is None:
            print(f"Detection failed or returned no detections for {image_path}")
            continue

        print(f"Detection successful for {image_path}. Detections:")
        for detection in detections.detections:
            print(f"Label: {detection.label}, Confidence: {detection.confidence}")


if __name__ == "__main__":
    main()
