"""
Intermediate worker that adds a random number to the input.
"""

import os
import pathlib

from modules.common.modules.logger import logger
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import add_random


def add_random_worker(
    seed: int,
    max_random_term: int,
    add_change_count: int,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    seed, max_random_term, and add_change_count are initial settings.
    input_queue and output_queue are the data queues.
    controller is how the main process communicates to this worker process.
    """
    # Instantiate logger
    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert local_logger is not None

    local_logger.info("Logger initialized", True)

    # Instantiate class object
    add_random_instance = add_random.AddRandom(
        seed, max_random_term, add_change_count, local_logger
    )

    # Loop forever until exit has been requested or sentinel value (consumer)
    while not controller.is_exit_requested():
        # Method blocks worker if pause has been requested
        controller.check_pause()

        # Get an item from the queue
        # If the queue is empty, the worker process will block
        # until the queue is non-empty
        term = input_queue.queue.get()

        # Exit on sentinel
        if term is None:
            break

        # All of the work should be done within the class
        # Getting the output is as easy as calling a single method
        # The class is reponsible for packing the intermediate type
        result, value = add_random_instance.run_add_random(term)

        # Check result
        if not result:
            continue

        # Put an item into the queue
        # If the queue is full, the worker process will block
        # until the queue is non-empty
        output_queue.queue.put(value)
