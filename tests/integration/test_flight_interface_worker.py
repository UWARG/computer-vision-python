"""
To test, start Mission Planner and forward MAVLink over TCP.
"""

import multiprocessing as mp
import queue
import time


from modules.flight_interface import flight_interface_worker
from modules import decision_command
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from modules.common.modules import position_global
from modules.common.modules.data_encoding import message_encoding_decoding
from modules.common.modules.data_encoding import worker_enum

MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_TIMEOUT = 10.0  # seconds
FLIGHT_INTERFACE_BAUD_RATE = 57600  # symbol rate
FLIGHT_INTERFACE_WORKER_PERIOD = 0.1  # seconds
WORK_COUNT = 4
COMMUNICATIONS_WORKER_ID = worker_enum.WorkerEnum.COMMUNICATIONS_WORKER


def simulate_communications_worker(
    in_queue: queue_proxy_wrapper.QueueProxyWrapper,
    data_point: position_global.PositionGlobal,
) -> None:
    """
    Encode coordinates and place into queue.
    """
    result, message = message_encoding_decoding.encode_position_global(
        COMMUNICATIONS_WORKER_ID, data_point
    )
    assert result
    assert message is not None

    in_queue.queue.put(message)

    return


def apply_decision_test(
    in_queue: queue_proxy_wrapper.QueueProxyWrapper,
    out_queue: queue_proxy_wrapper.QueueProxyWrapper,
) -> bool:
    """
    Test the apply_decision method by sending DecisionCommands via input_queue and verifying odometry data from output_queue.
    """
    # Test MOVE_TO_RELATIVE_POSITION command
    decision_command_move_relative = (
        decision_command.DecisionCommand.create_move_to_relative_position_command(
            15.0, 0.0, -5.0  # Move 10 meters north, 0 meters east, ascend 5 meters
        )
    )

    print("Applying MOVE_TO_RELATIVE_POSITION command...")
    in_queue.queue.put(decision_command_move_relative)

    # Wait for the drone to move
    time.sleep(15)

    # Verify the drone's position
    try:
        odometry_and_time_data = out_queue.queue.get_nowait()
        print("Drone position after MOVE_TO_RELATIVE_POSITION command:")
        print(odometry_and_time_data.odometry_data.position)
    except queue.Empty:
        print("No odometry data received after MOVE_TO_RELATIVE_POSITION command.")
        return False

    # Test MOVE_TO_ABSOLUTE_POSITION command
    decision_command_move_absolute = (
        decision_command.DecisionCommand.create_move_to_absolute_position_command(
            43.4337659, -80.5769169, 400  # Somewhere at WRESTRC
        )
    )

    print("Applying MOVE_TO_ABSOLUTE_POSITION command...")
    in_queue.queue.put(decision_command_move_absolute)

    # Wait for the drone to move
    time.sleep(15)

    # Verify the drone's position
    try:
        odometry_and_time_data = out_queue.queue.get_nowait()
        print("Drone position after MOVE_TO_ABSOLUTE_POSITION command:")
        print(odometry_and_time_data.odometry_data.position)
    except queue.Empty:
        print("No odometry data received after MOVE_TO_ABSOLUTE_POSITION command.")
        return False

    # Test LAND_AT_RELATIVE_POSITION command
    decision_command_land_relative = (
        decision_command.DecisionCommand.create_land_at_relative_position_command(-20, 10, 0)
    )

    print("Applying LAND_AT_RELATIVE_POSITION command...")
    in_queue.queue.put(decision_command_land_relative)

    # Wait for the drone to land
    time.sleep(15)

    # Verify that the drone has landed
    try:
        odometry_and_time_data = out_queue.queue.get_nowait()
        print("Drone position after LAND_AT_RELATIVE_POSITION command:")
        print(odometry_and_time_data.odometry_data.position)
    except queue.Empty:
        print("No odometry data received after LAND_AT_RELATIVE_POSITION command.")
        return False

    print("apply_decision tests completed successfully.")
    return True


def main() -> int:
    """
    Main function.
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    home_position_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    communications_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    communications_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=flight_interface_worker.flight_interface_worker,
        args=(
            MAVLINK_CONNECTION_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_BAUD_RATE,
            FLIGHT_INTERFACE_WORKER_PERIOD,
            in_queue,  # Added input_queue
            out_queue,
            home_position_out_queue,
            communications_in_queue,
            controller,
        ),
    )

    # Run
    worker.start()

    time.sleep(3)

    # Test
    home_position = home_position_out_queue.queue.get()
    assert home_position is not None

    data_points = [position_global.PositionGlobal.create(43.471468, -80.544205, 335), 
                   position_global.PositionGlobal.create(43.6629, -79.3957, 105),
                   position_global.PositionGlobal.create(43.2609, -79.9192, 100),
                   position_global.PositionGlobal.create(43.7735, -79.5019, 170)
                   ]

    # Simulate communications worker
    for i in range(0, WORK_COUNT):
        simulate_communications_worker(communications_in_queue, home_position, data_points[i])

    # Test flight interface worker sending statustext messages
    for i in range(0, WORK_COUNT):
        try:
            input_data: bytes = communications_out_queue.queue.get_nowait()
            assert input_data is not None
        except queue.Empty:
            print("Output queue has no more messages to process, exiting")
            break

    assert communications_out_queue.queue.empty()

    # Run the apply_decision tests
    test_result = apply_decision_test(in_queue, out_queue)
    if not test_result:
        print("apply_decision tests failed.")
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
