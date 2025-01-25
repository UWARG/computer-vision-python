"""
Gets odometry information from drone.
"""

import os
import pathlib
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import flight_interface
from ..common.modules.logger import logger
from ..common.modules.mavlink import flight_controller
from ..common.modules.data_encoding import message_encoding_decoding
from ..common.modules.data_encoding import metadata_encoding_decoding
from ..common.modules import position_global


def flight_interface_worker(
    address: str,
    timeout: float,
    baud_rate: int,
    period: float,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    communications_output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    coordinates_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    address, timeout is initial setting.
    period is minimum period between loops.
    input_queue and output_queue are the data queues.
    communications_output_queue is a one time queue that sends communications worker the home address.
    coordinates_input_queue provides the flight interface worker with a list of GPS coordinates.
    controller is how the main process communicates to this worker process.
    """
    # TODO: Error handling

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert local_logger is not None

    local_logger.info("Logger initialized", True)

    result, interface = flight_interface.FlightInterface.create(
        address, timeout, baud_rate, local_logger
    )
    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert interface is not None

    home_position = interface.get_home_position()
    communications_output_queue.queue.put(home_position)

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        coordinate = coordinates_input_queue.queue.get()
        if coordinate is None:
            local_logger.info("Received type None, exiting")
            break

        if not isinstance(coordinate, bytes):
            local_logger.warning(f"Skipping unexpected input: {coordinate}")
            continue

        result, value = interface.run(coordinate)
        if not result:
            continue

        output_queue.queue.put(value)

        # Check for decision commands
        if not input_queue.queue.empty():
            command = input_queue.queue.get()
            # Pass the decision command to the flight controller
            interface.apply_decision(command)
