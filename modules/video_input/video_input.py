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
        self.device = camera_device.CameraDevice(camera_name)

    def run(self):
        """
        Returns a possible FrameAndTime with current timestamp
        """
        result, image = self.device.get_image()
        if not result:
            return False, None

        return True, frame_and_time.FrameAndTime(image)

# pylint: enable=too-few-public-methods
