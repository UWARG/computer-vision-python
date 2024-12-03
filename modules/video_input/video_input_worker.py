"""
Gets images from the camera.
"""

import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import video_input


def video_input_worker(
    camera_name: "int | str",
    period: float,
    width: int,
    height: int,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    camera_name is initial setting.
    period is minimum period between loops.
    width is the width of the images the camera takes in pixels
    height is the height of the images the camera takes in pixelss
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    input_device = video_input.VideoInput(camera_name, width, height)

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)
