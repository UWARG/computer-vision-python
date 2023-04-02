"""
Ending worker that concatenates a prefix and suffix and then prints the result.
"""
import multiprocessing as mp

from utilities import manage_worker
from . import concatenator


def concatenator_worker(prefix: str, suffix: str,
                        input_queue: mp.Queue, worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    prefix and suffix are initial settings.
    input_queue is the data queue.
    worker_manager is how the main process communicates to this worker process.
    """
    # Instantiate class object
    concatenator_instance = concatenator.Concatenator(prefix, suffix)

    # Loop forever until exit has been requested
    while not worker_manager.is_exit_requested():
        # Method blocks worker if pause has been requested
        worker_manager.check_pause()

        # Get an item from the queue
        # If the queue is empty, the worker process will block
        # until the queue is non-empty
        input_data = input_queue.get()

        # When everything is exiting, the queue will be filled with None objects
        if input_data is None:
            continue

        # All of the work should be done within the class
        # Getting the output is as easy as calling a single method
        # The class is reponsible for unpacking the intermediate type
        result, value = concatenator_instance.run_concatenation(input_data)

        # Check result
        if not result:
            continue

        # Print the string
        print(value)
