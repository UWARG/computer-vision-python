"""
Simple hardware test, requires camera.
"""

import pathlib

from modules.common.modules.camera import camera_factory
from modules.common.modules.camera import camera_opencv
from modules.common.modules.camera import camera_picamera2
from modules.common.modules.logger import logger
from modules.video_input import video_input


# Modify as needed
CAMERA = camera_factory.CameraOption.PICAM2
WIDTH = 640
HEIGHT = 480
CONFIG = camera_picamera2.ConfigPiCamera2()
IMAGE_NAME = None  # Not saving any pictures


def main() -> int:
    """
    Main function.
    """
    # Logger
    test_name = pathlib.Path(__file__).stem
    result, local_logger = logger.Logger.create(test_name, False)
    assert result
    assert local_logger is not None

    # Setup
    result, camera = video_input.VideoInput.create(
        CAMERA, WIDTH, HEIGHT, CONFIG, IMAGE_NAME, local_logger
    )
    assert result
    assert camera is not None

    # Run
    result, image = camera.run()

    # Test
    assert result
    assert image is not None

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
