"""
Combines image and timestamp together.
"""

from ..common.camera.modules import camera_device
from .. import image_and_time


class VideoInput:
    """
    Combines image and timestamp together.
    """

    def __init__(self, camera_name: "int | str", save_name: str = ""):
        self.device = camera_device.CameraDevice(camera_name, 1, save_name)

    def run(self) -> "tuple[bool, image_and_time.ImageAndTime | None]":
        """
        Returns a possible ImageAndTime with current timestamp.
        """
        result, image = self.device.get_image()
        if not result:
            return False, None

        return image_and_time.ImageAndTime.create(image)
