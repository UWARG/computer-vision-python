"""
Combines image and timestamp together.
"""

from .. import image_and_time
from ..common.modules.camera import camera_device


class VideoInput:
    """
    Combines image and timestamp together.
    """

    def __init__(
        self, camera_name: "int | str", save_name: str = "", use_picamera: bool = False
    ) -> None:
        self.device = camera_device.CameraDevice(use_picamera, camera_name, 1, save_name)

    def run(self) -> "tuple[bool, image_and_time.ImageAndTime | None]":
        """
        Returns a possible ImageAndTime with current timestamp.
        """
        result, image = self.device.get_image()
        if not result:
            return False, None

        return image_and_time.ImageAndTime.create(image)
