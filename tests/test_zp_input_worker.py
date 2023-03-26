"""
Tests process
"""
import multiprocessing as mp
import queue
import time

from utilities import manage_worker
from modules.zp_input import zp_input_worker

PORT="/dev/ttyS0"
BAUDRATE=115_200


if __name__ == "__main__":
    # Setup
    process_manager = manage_worker.ManageWorker()
    out_queue = mp.Queue()

    worker = mp.Process(
        target=zp_input_worker.zp_input_worker,
        args=(PORT, BAUDRATE, out_queue, process_manager)
    )

    # Run
    worker.start()

    time.sleep(3)

    process_manager.request_exit()

    # Test
    while True:
        try:
            input_data = out_queue.get_nowait()
            assert str(type(input_data)) == \
                "<class \'modules.telemetry_and_time.TelemetryAndTime\'>"
            print(type(input_data))

        except queue.Empty:
            break

    print("Done!")
