"""
Intermediate worker that adds a random number to the input.
"""
import queue

from utilities import manage_worker
from . import add_random


# As kwargs is not being used, this function needs many parameters
# pylint: disable=too-many-arguments
def add_random_worker(seed: int,
                      max_random_term: int,
                      add_change_count: int,
                      input_queue: queue.Queue,
                      output_queue: queue.Queue,
                      worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    seed, max_random_term, and add_change_count are initial settings.
    input_queue and output_queue are the data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    # Instantiate class object
    add_random_instance = add_random.AddRandom(seed, max_random_term, add_change_count)

    # Loop forever until exit has been requested
    while not worker_manager.is_exit_requested():
        # Method blocks worker if pause has been requested
        worker_manager.check_pause()

        # Get an item from the queue
        # If the queue is empty, the worker process will block
        # until the queue is non-empty
        term = input_queue.get()

        # When everything is exiting, the queue will be filled with None objects
        if term is None:
            continue

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
        output_queue.put(value)

# pylint: enable=too-many-arguments
