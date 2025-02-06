"""
Logs data and forwards it.
"""

import os
import pathlib
import queue
import time

from modules import object_in_world
from . import communications
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from ..common.modules.logger import logger


def communications_worker(
    timeout: float,
    period: float,
    home_position_queue: queue_proxy_wrapper.QueueProxyWrapper,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    message_output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    home_position_queue contains home positions for creating communications object.
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

    # Get home position
    try:
        home_position = home_position_queue.queue.get(timeout=timeout)
    except queue.Empty:
        local_logger.error("Home position queue timed out on startup", True)
        return

    local_logger.info(f"Home position received: {home_position}", True)

    result, comm = communications.Communications.create(home_position, local_logger)
    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert comm is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()

        if input_data is None:
            local_logger.info("Recieved type None, exiting.")
            break

        is_invalid = False

        for single_input in input_data:
            if not isinstance(single_input, object_in_world.ObjectInWorld):
                local_logger.warning(
                    f"Skipping unexpected input: {input_data}, because of unexpected value: {single_input}"
                )
                is_invalid = True
                break

        if is_invalid:
            continue

        result, metadata, list_of_messages = comm.run(input_data)
        if not result:
            continue

        output_queue.queue.put(metadata)
        message_output_queue.queue.put(metadata)

        for message in list_of_messages:

            time.sleep(period)

            output_queue.queue.put(message)
            message_output_queue.queue.put(message)
