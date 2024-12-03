"""
Combines image and timestamp together.
"""

from .. import image_and_time
from ..common.modules.camera import camera_factory


class VideoInput:
    """
    Combines image and timestamp together.
    """

    def __init__(self, camera_option: int, width: int, height: int) -> None:
        self.device = camera_factory.create_camera(camera_option, width, height)

    def run(self) -> "tuple[bool, image_and_time.ImageAndTime | None]":
        """
        Returns a possible ImageAndTime with current timestamp.
        """
        result, image = self.device.run()
        if not result:
            return False, None

        return image_and_time.ImageAndTime.create(image)
