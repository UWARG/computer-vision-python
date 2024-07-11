"""
To test, start Mission Planner and forward MAVLink over TCP.
"""

import multiprocessing as mp
import queue
import time

from modules import odometry_and_time
from modules.flight_interface import flight_interface_worker
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_TIMEOUT = 10.0  # seconds
FLIGHT_INTERFACE_BAUD_RATE = 57600  # symbol rate
FLIGHT_INTERFACE_WORKER_PERIOD = 0.1  # seconds


def main() -> int:
    """
    Main function.
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=flight_interface_worker.flight_interface_worker,
        args=(
            MAVLINK_CONNECTION_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_BAUD_RATE,
            FLIGHT_INTERFACE_WORKER_PERIOD,
            out_queue,
            controller,
        ),
    )

    # Run
    worker.start()

    time.sleep(3)

    controller.request_exit()

    # Test
    while True:
        try:
            input_data: odometry_and_time.OdometryAndTime = out_queue.queue.get_nowait()
            assert str(type(input_data)) == "<class 'modules.odometry_and_time.OdometryAndTime'>"
            assert input_data.odometry_data is not None

        except queue.Empty:
            break

    # Teardown
    worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
