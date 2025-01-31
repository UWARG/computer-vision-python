"""
Test MAVLink integration test 
"""

import multiprocessing as mp
import queue
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller

from modules.common.modules import position_global
from modules.flight_interface import flight_interface_worker


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_TIMEOUT = 10.0  # seconds
FLIGHT_INTERFACE_BAUD_RATE = 57600  # symbol rate
FLIGHT_INTERFACE_WORKER_PERIOD = 0.1  # seconds


def apply_communications_test(
    in_queue: queue_proxy_wrapper.QueueProxyWrapper,
    out_queue: queue_proxy_wrapper.QueueProxyWrapper,
) -> None:
    """
    Method to send in hardcoded GPS coordinates to the flight interface worker
    """
    gps_coordinates = [
        position_global.PositionGlobal.create(43.47321268948186, -80.53950244232878, 10),  # E7
        position_global.PositionGlobal.create(37.7749, 122.4194, 30),  # San Francisco
        position_global.PositionGlobal.create(40.7128, 74.0060, -5.6),  # New York
        position_global.PositionGlobal.create(51.5072, 0.1276, 20.1),  # London UK
    ]

    # Place the GPS coordinates
    for success, gps_coordinate in gps_coordinates:
        if not success:
            return False
        in_queue.queue.put(gps_coordinate)

    # Wait for processing
    time.sleep(10)

    # Verify that stuff is sending
    try:
        # pylint: disable=unused-variable
        for i in range(len(gps_coordinates)):
            data = out_queue.queue.get_nowait()
            print(f"MAVLink data sent by drone: {data}")
    except queue.Empty:
        print("Output queue is empty.")
        return False

    print("apply_communications_test completed successfully")
    return True


# pylint: disable=duplicate-code
def main() -> int:
    """
    Main function
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    home_position_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=flight_interface_worker.flight_interface_worker,
        args=(
            MAVLINK_CONNECTION_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_BAUD_RATE,
            FLIGHT_INTERFACE_WORKER_PERIOD,
            in_queue,
            out_queue,
            home_position_out_queue,
            controller,
        ),
    )

    worker.start()

    time.sleep(3)

    # Test
    home_position = home_position_out_queue.queue.get()
    assert home_position is not None

    # Run the apply_communication tests
    test_result = apply_communications_test(in_queue, out_queue)
    if not test_result:
        print("apply_communications test failed.")
        worker.terminate()
        return -1

    # Teardown
    controller.request_exit()
    worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
