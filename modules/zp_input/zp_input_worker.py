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
                    output_queue: mp.Queue,
                    main_control: manage_worker.ManageWorker):
    """
    Worker process.

    port is UART port.
    baudrate is UART baudrate.
    output_queue is the data queue.
    main_control is how the main process communicates to this worker process.
    """
    input_device = zp_input.TelemetryInput(port, baudrate)

    while not main_control.is_exit_requested():
        main_control.check_pause()

        result, value = input_device.run()
        if result:
            output_queue.put(value)
