"""
Gets frames and adds a timestamp
"""
import queue
import time

from utilities import manage_worker
from . import video_input


def video_input_worker(camera_name: "int | str",
                       period: float,
                       output_queue: queue.Queue,
                       worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    camera_name is initial setting.
    period is minimum period between loops.
    output_queue is the data queue.
    worker_manager is how the main process communicates to this worker process.
    """
    # TODO: Logging?
    input_device = video_input.VideoInput(camera_name)

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.put(value)
