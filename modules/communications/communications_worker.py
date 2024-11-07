"""
Logs data and forwards it.
"""

import os
import pathlib

from . import communications
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from ..common.modules.logger import logger


def communications_worker(
    home_position_queue: queue_proxy_wrapper.QueueProxyWrapper,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    home_position: get home_position for init
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

    # Get home location
    home_position = home_position_queue.queue.get()

    result, comm = communications.Communications.create(home_position, local_logger)
    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert comm is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        result, value = comm.run(input_queue.queue.get())
        if not result:
            continue

        output_queue.queue.put(value)
