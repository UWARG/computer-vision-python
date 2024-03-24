"""
Gets images from the camera.
"""

import inspect
import queue
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from ..multiprocess_logging import multiprocess_logging
from . import video_input


def video_input_worker(
    camera_name: "int | str",
    period: float,
    save_name: str,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    logging_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    camera_name is initial setting.
    period is minimum period between loops.
    save_name is path for logging.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    input_device = video_input.VideoInput(logging_queue, camera_name, save_name)

    try:
        frame = inspect.currentframe()
        multiprocess_logging.log_message(
            "video_input started", multiprocess_logging.DEBUG, frame, logging_queue
        )
        # logging_queue.queue.put((multiprocess_logging.message_and_metadata('video_input started', frame), logging.DEBUG), block=False)
    except queue.Full:
        pass

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)
