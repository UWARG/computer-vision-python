"""
Auto-landing worker.
"""

import pathlib
import os

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import auto_landing
from ..common.modules.logger import logger


def auto_landing_worker(
    fov_x: float,
    fov_y: float,
    im_h: float,
    im_w: float,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    input_queue and output_queue are data queues.
    controller is how the main process communicates to this worker process.
    """

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert local_logger is not None

    local_logger.info("Logger initialized", True)

    result, auto_lander = auto_landing.AutoLanding.create(fov_x, fov_y, im_h, im_w, local_logger)

    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert auto_lander is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            continue

        result, value = auto_lander.run(input_data)
        if not result:
            continue

        output_queue.queue.put(value)
