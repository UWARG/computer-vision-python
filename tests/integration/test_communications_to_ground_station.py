"""
Test MAVLink integration test 
"""

import multiprocessing as mp
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller

from modules.common.modules import position_global
from modules.flight_interface import flight_interface_worker


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_TIMEOUT = 30.0  # seconds
FLIGHT_INTERFACE_BAUD_RATE = 57600  # symbol rate
FLIGHT_INTERFACE_WORKER_PERIOD = 0.1  # seconds


def apply_communications_test(
    communications_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
) -> bool:
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
        communications_input_queue.queue.put(gps_coordinate)

    # Wait for processing
    time.sleep(10)

    # Verify that stuff is sending
    print(
        "TEST OPERATOR ACTION REQUIRED: Open mission planner's MAVLink inspector or the groundside repo (https://github.com/UWARG/statustext-parser-2025) to check for MAVLink messages"
    )
    return True


# pylint: disable=duplicate-code
def main() -> int:
    """
    Main function
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    communications_input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=flight_interface_worker.flight_interface_worker,
        args=(
            MAVLINK_CONNECTION_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_BAUD_RATE,
            FLIGHT_INTERFACE_WORKER_PERIOD,
            in_queue,
            out_queue,
            communications_input_queue,
            controller,
        ),
    )

    worker.start()

    time.sleep(3)

    # Test
    home_position = communications_input_queue.queue.get()
    assert home_position is not None

    # Run the apply_communication tests
    test_result = apply_communications_test(communications_input_queue)
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
