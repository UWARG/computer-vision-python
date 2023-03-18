"""
Gets frames and adds a timestamp
"""

import multiprocessing as mp

from . import video_input
# Import required beyond the current directory
# pylint: disable=import-error
from utilities import manage_worker
# pylint: enable=import-error


def video_input_worker(camera_name: "int | str",
                       output_queue: mp.Queue,
                       main_control: manage_worker.ManageWorker):
    """
    Worker process.

    camera_name is initial setting.
    output_queue is the data queue.
    main_control is how the main process communicates to this worker process.
    """
    input_device = video_input.VideoInput(camera_name)

    while not main_control.is_exit_requested():
        main_control.check_pause()

        result, value = input_device.run()
        if result:
            output_queue.put(value)
