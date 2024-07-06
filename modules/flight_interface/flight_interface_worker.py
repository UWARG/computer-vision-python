"""
Gets odometry information from drone.
"""

import inspect
import os
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import flight_interface
from ..logger import logger


def flight_interface_worker(
    address: str,
    timeout: float,
    baud_rate: int,
    period: float,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    address, timeout is initial setting.
    period is minimum period between loops.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    # TODO: Error handling

    process_id = os.getpid()
    result, flight_interface_logger = logger.Logger.create(f"flight_interface_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert flight_interface_logger is not None

    frame = inspect.currentframe()
    flight_interface_logger.info("Flight interface logger initialized", frame)

    result, interface = flight_interface.FlightInterface.create(
        address, timeout, baud_rate, flight_interface_logger
    )
    if not result:
        frame = inspect.currentframe()
        flight_interface_logger.error("Worker failed to create class object", frame)
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
