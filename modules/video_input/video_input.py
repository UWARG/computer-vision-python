"""
Combines image and timestamp together.
"""

import inspect
import queue

from utilities.workers import queue_proxy_wrapper
from ..common.camera.modules import camera_device
from ..multiprocess_logging import multiprocess_logging
from .. import image_and_time


class VideoInput:
    """
    Combines image and timestamp together.
    """

    def __init__(
        self,
        logging_queue: queue_proxy_wrapper.QueueProxyWrapper,
        camera_name: "int | str",
        save_name: str = "",
    ) -> None:
        self.device = camera_device.CameraDevice(camera_name, 1, save_name)
        self.logging_queue = logging_queue

    def run(self) -> "tuple[bool, image_and_time.ImageAndTime | None]":
        """
        Returns a possible ImageAndTime with current timestamp.
        """
        result, image = self.device.get_image()

        try:
            frame = inspect.currentframe()
            multiprocess_logging.log_message(
                f"image size {image.shape}", multiprocess_logging.DEBUG, frame, self.logging_queue
            )
            # self.logging_queue.queue.put((multiprocess_logging.message_and_metadata(f'image size {image.shape}', frame), logging.DEBUG), block=False)
        except queue.Full:
            pass

        if not result:
            return False, None

        return image_and_time.ImageAndTime.create(image)
