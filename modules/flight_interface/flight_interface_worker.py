"""
Gets odometry information from drone.
"""
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import flight_interface


def flight_interface_worker(address: str,
                            timeout_home: float,
                            period: float,
                            output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                            controller: worker_controller.WorkerController):
    """
    Worker process. 

    address, timeout_home is initial setting.
    period is minimum period between loops.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    result, interface = flight_interface.FlightInterface.create(address, timeout_home)
    if not result:
        return

    # Get Pylance to stop complaining
    assert interface is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = interface.run()
        if not result:
            continue

        output_queue.queue.put(value)
