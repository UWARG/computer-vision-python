"""
Test worker process.
"""
import multiprocessing as mp
import queue
import time

from modules import odometry_and_time
from modules.flight_interface import flight_interface_worker
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_WORKER_PERIOD = 0.1  # seconds
TIMEOUT_HOME = 10.0  # seconds


# To test, start Mission Planner and forward MAVLink over TCP
if __name__ == "__main__":
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=flight_interface_worker.flight_interface_worker,
        args=(
            MAVLINK_CONNECTION_ADDRESS,
            TIMEOUT_HOME,
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
            assert str(type(input_data)) == "<class \'modules.odometry_and_time.OdometryAndTime\'>"
            assert input_data.odometry_data is not None

        except queue.Empty:
            break

    # Teardown
    worker.join()

    print("Done!")
