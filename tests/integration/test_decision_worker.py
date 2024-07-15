"""
Test worker process.
"""

import multiprocessing as mp
import time

from modules.decision import decision_worker
from modules import drone_odometry_local
from modules.cluster_estimation import cluster_estimation
from modules import odometry_and_time
from modules.decision import decision
from utilities.workers import worker_controller
from utilities.workers import queue_proxy_wrapper


DISTANCE_SQUARED_THRESHOLD = 25  # squared meters

TOLERANCE = 0.1  # meters

CAMERA_FOV_FORWARDS = 90  # degrees
CAMERA_FOV_SIDEWAYS = 120  # degrees

SEARCH_HEIGHT = 10  # meters
SEARCH_OVERLAP = 0.2

SMALL_ADJUSTMENT = 0.5  # meters


def simulate_cluster_estimation_worker(
    min_activation_threshold: int,
    min_new_points_to_run: int,
    random_state: int,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
) -> None:
    """
    Places list of detected landing pads into the queue.
    """

    result, estimator = cluster_estimation.ClusterEstimation.create(
        min_activation_threshold, min_new_points_to_run, random_state
    )
    assert result
    assert estimator is not None

    result, cluster_pads = estimator.run(input_queue, False)
    assert result
    assert cluster_pads is not None

    input_queue.queue.put(cluster_pads)


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


def main():
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

    # Simulate odometry data
    for i in range(1, 5):
        simulate_flight_interface_worker(i, odometry_input_queue)

    # Simulate cluster estimation
    simulate_cluster_estimation_worker(1, 1, 1, cluster_input_queue)

    time.sleep(1)

    controller.request_exit()

    # Test
    try:
        while True:
            decision_output: decision.Decision = decision_output_queue.queue.get_nowait()
            print(f"Decision output: {decision_output}")
            assert decision_output is not None
    except queue_proxy_wrapper.queue.Empty:
        pass

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
