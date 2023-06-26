"""
Combines image and timestamp together
"""

from ..common.camera.modules import camera_device
from .. import frame_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class VideoInput:
    """
    Combines image and timestamp together
    """

    def __init__(self, camera_name: "int | str"):
        # TODO: Logging?
        self.device = camera_device.CameraDevice(camera_name)

    def run(self) -> "tuple[bool, frame_and_time.FrameAndTime | None]":
        """
        Returns a possible FrameAndTime with current timestamp
        """
        result, image = self.device.get_image()
        if not result:
            return False, None

        return frame_and_time.FrameAndTime.create(image)

# pylint: enable=too-few-public-methods
