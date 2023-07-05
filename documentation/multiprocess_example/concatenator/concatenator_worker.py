"""
Ending worker that concatenates a prefix and suffix and then prints the result.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import concatenator


def concatenator_worker(prefix: str,
                        suffix: str,
                        input_queue: queue_proxy_wrapper.QueueProxyWrapper,
                        controller: worker_controller.WorkerController):
    """
    Worker process.

    prefix and suffix are initial settings.
    input_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    # Instantiate class object
    concatenator_instance = concatenator.Concatenator(prefix, suffix)

    # Loop forever until sentinel value (consumer)
    while True:
        # Method blocks worker if pause has been requested
        controller.check_pause()

        # Get an item from the queue
        # If the queue is empty, the worker process will block
        # until the queue is non-empty
        input_data = input_queue.queue.get()

        # Exit on sentinel
        if input_data is None:
            break

        # All of the work should be done within the class
        # Getting the output is as easy as calling a single method
        # The class is reponsible for unpacking the intermediate type
        result, value = concatenator_instance.run_concatenation(input_data)

        # Check result
        if not result:
            continue

        # Print the string
        print(value)
