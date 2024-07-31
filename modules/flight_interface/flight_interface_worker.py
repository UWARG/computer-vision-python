"""
Gets odometry information from drone.
"""

import inspect
import os
import pathlib
import time
import multiprocessing as mp

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
    odometry_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    address, timeout is initial setting.
    period is minimum period between loops.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    if len(odometry_queue) > 1:
        print("ERROR: Queue should have a maximum size of 1")
        return

    # TODO: Error handling

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert local_logger is not None

    frame = inspect.currentframe()
    local_logger.info("Logger initialized", frame)

    result, interface = flight_interface.FlightInterface.create(
        address, timeout, baud_rate, local_logger
    )
    if not result:
        frame = inspect.currentframe()
        local_logger.error("Worker failed to create class object", frame)
        return

    # Get Pylance to stop complaining
    assert interface is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = interface.run()
        if not result:
            continue

        # Replace any existing odometry data with the latest odometry data
        try:
            odometry_queue.queue.get_nowait()
        except queue_proxy_wrapper.queue.Empty:
            pass

        odometry_queue.queue.put(value)
        output_queue.queue.put(value)
