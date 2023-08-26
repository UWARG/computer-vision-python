import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import flight_input

def flight_input_worker(address: str,
                        period: float,
                        output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                        controller: worker_controller.WorkerController):
    """
    TODO: is save_name required for this worker?
    Worker process. 

    address is initial setting.
    period is minimum period between loops.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    input_device = flight_input.FlightInput(address)

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)
