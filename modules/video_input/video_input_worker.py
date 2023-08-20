"""
Gets images from the camera.
"""
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import video_input


def video_input_worker(camera_name: "int | str",
                       period: float,
                       save_name: str,
                       output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                       controller: worker_controller.WorkerController):
    """
    Worker process.

    camera_name is initial setting.
    period is minimum period between loops.
    save_name is path for logging.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    input_device = video_input.VideoInput(camera_name, save_name)

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)
