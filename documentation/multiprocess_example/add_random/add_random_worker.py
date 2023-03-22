"""
Intermediate worker that adds a random number to the input.
"""
import multiprocessing as mp

from utilities import manage_worker
from . import add_random


# As kwargs is not being used, this function needs many parameters
# pylint: disable=too-many-arguments
def add_random_worker(seed: int, max_random_term: int, add_change_count: int,
                      input_queue: mp.Queue, output_queue: mp.Queue,
                      main_control: manage_worker.ManageWorker):
    """
    Worker process.

    main_control is how the main process communicates to this worker process.
    seed, max_random_term, and add_change_count are initial settings.
    input_queue and output_queue are the data queues.
    """
    # Instantiate class object
    add_random_instance = add_random.AddRandom(seed, max_random_term, add_change_count)

    # Loop forever until exit has been requested
    while not main_control.is_exit_requested():
        # Method blocks worker if pause has been requested
        main_control.check_pause()

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
        if result:
            # Put an item into the queue
            # If the queue is full, the worker process will block
            # until the queue is non-empty
            output_queue.put(value)

# pylint: enable=too-many-arguments
