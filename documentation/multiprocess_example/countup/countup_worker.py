"""
Beginning worker that counts up from a starting value.
"""
import multiprocessing as mp

from utilities import manage_worker
from . import countup


def countup_worker(start_thousands: int, max_iterations: int,
                   output_queue: mp.Queue, worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    start_thousands and max_iterations are initial settings.
    start_thousands is the start value (in thousands),
    while max_iterations is the maximum value that the counter will reach before resetting.
    output_queue is the data queue.
    worker_manager is how the main process communicates to this worker process.
    """
    # Instantiate class object
    countup_instance = countup.Countup(start_thousands, max_iterations)

    # Loop forever until exit has been requested
    while not worker_manager.is_exit_requested():
        # Method blocks worker if pause has been requested
        worker_manager.check_pause()

        # All of the work should be done within the class
        # Getting the output is as easy as calling a single method
        result, value = countup_instance.run_countup()

        # Check result
        if not result:
            continue

        # Put an item into the queue
        # If the queue is full, the worker process will block
        # until the queue is non-empty
        output_queue.put(value)
