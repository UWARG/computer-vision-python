"""
Gets frames and adds a timestamp
"""

import multiprocessing as mp

from . import zp_input
# Import required beyond the current directory
# pylint: disable=import-error
from utilities import manage_worker
# pylint: enable=import-error


def zp_input_worker(port: str, baudrate: int,
                    telemetry_output_queue: mp.Queue, request_output_queue: mp.Queue,
                    worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    port is UART port.
    baudrate is UART baudrate.
    telemetry_output_queue is the telemetry queue.
    request_output_queue is the ZP request queue.
    worker_manager is how the main process communicates to this worker process.
    """
    input_device = zp_input.ZpInput(port, baudrate)

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        result, value = input_device.run()
        if not result:
            continue

        # Get Pylance to stop complaining
        assert value is not None
        assert value.message is not None

        # Decide which worker to send to next depending on message type
        if value.message.header.type == 0:
            # Odometry
            telemetry_output_queue.put(value)
        elif value.message.header.type == 1:
            # Request
            request_output_queue.put(value)
        else:
            # TODO: Invalid type, log it?
            pass
