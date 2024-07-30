"""
Test worker process.
"""

import multiprocessing as mp
import time

from modules import drone_odometry_local
from modules import object_in_world
from modules import odometry_and_time
from modules.cluster_estimation import cluster_estimation
from modules.decision import decision
from modules.decision import decision_worker
from utilities.workers import worker_controller
from utilities.workers import queue_proxy_wrapper


DISTANCE_SQUARED_THRESHOLD = 25  # Distance is in meters

TOLERANCE = 0.1  # meters

CAMERA_FOV_FORWARDS = 90  # degrees
CAMERA_FOV_SIDEWAYS = 120  # degrees

SEARCH_HEIGHT = 10  # meters
SEARCH_OVERLAP = 0.2

SMALL_ADJUSTMENT = 0.5  # meters

WORK_COUNT = 5


def simulate_cluster_estimation_worker(
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
) -> None:
    """
    Places list of detected landing pads into the queue.
    """

    fake_objects = [
        object_in_world.ObjectInWorld.create(1.0, 2.0, 0.9)[1],
        object_in_world.ObjectInWorld.create(4.5, 3.0, 2.0)[1],
        object_in_world.ObjectInWorld.create(7.2, 2.9, 4.6)[1],
    ]

    for obj in fake_objects:
        assert obj is not None

    input_queue.queue.put(fake_objects)


def simulate_flight_interface_worker(
    timestamp: float, odometry_queue: queue_proxy_wrapper.QueueProxyWrapper
) -> None:
    """
    Place odometry data into queue of size 1.
    """

    result, drone_position = drone_odometry_local.DronePositionLocal.create(0.0, 2.0, -1.0)
    assert result
    assert drone_position is not None

    result, drone_orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)
    assert result
    assert drone_orientation is not None

    result, drone_odometry = drone_odometry_local.DroneOdometryLocal.create(
        drone_position, drone_orientation
    )
    assert result
    assert drone_odometry is not None

    result, drone_odometry_and_time = odometry_and_time.OdometryAndTime.create(drone_odometry)
    assert result
    assert drone_odometry_and_time is not None

    drone_odometry_and_time.timestamp = timestamp

    odometry_queue.queue.put(drone_odometry_and_time)


def main() -> int:
    """
    Main function.
    """
    controller = worker_controller.WorkerController()
    mp_manager = mp.Manager()

    cluster_input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    odometry_input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager, maxsize=1)
    decision_output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=decision_worker.decision_worker,
        args=(
            DISTANCE_SQUARED_THRESHOLD,
            TOLERANCE,
            CAMERA_FOV_FORWARDS,
            CAMERA_FOV_SIDEWAYS,
            SEARCH_HEIGHT,
            SEARCH_OVERLAP,
            SMALL_ADJUSTMENT,
            odometry_input_queue,
            cluster_input_queue,
            decision_output_queue,
            controller,
        ),
    )

    # Starts the decision worker
    worker.start()

    # Simulate odometry data and cluster estimation
    for i in range(0, WORK_COUNT):
        simulate_flight_interface_worker(i, odometry_input_queue)
        simulate_cluster_estimation_worker(cluster_input_queue)

    time.sleep(1)

    for i in range(0, WORK_COUNT):
        simulate_flight_interface_worker(i, odometry_input_queue)
        simulate_cluster_estimation_worker(cluster_input_queue)

    controller.request_exit()

    # Test
    for i in range(0, WORK_COUNT * 2):
        decision_output: decision.Decision = decision_output_queue.queue.get_nowait()
        print(f"Decision output: {decision_output}")
        assert decision_output is not None

    # Teardown
    odometry_input_queue.fill_and_drain_queue()
    cluster_input_queue.fill_and_drain_queue()
    worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
