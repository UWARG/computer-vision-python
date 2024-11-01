"""
Test worker process.
"""

import multiprocessing as mp
import time

import numpy as np

from modules import detections_and_time
from modules import merged_odometry_detections
from modules import odometry_and_time
from modules.common.mavlink.modules import drone_odometry_local
from modules.data_merge import data_merge_worker
from utilities.workers import queue_proxy_wrapper, worker_controller

DATA_MERGE_WORKER_TIMEOUT = 10.0  # seconds


def simulate_detect_target_worker(
    timestamp: float, detections_queue: queue_proxy_wrapper.QueueProxyWrapper
) -> None:
    """
    Place the detection into the queue.
    """
    result, detections = detections_and_time.DetectionsAndTime.create(timestamp)
    assert result
    assert detections is not None

    result, detection = detections_and_time.Detection.create(np.array([0.0, 0.0, 1.0, 1.0]), 0, 1.0)
    assert result
    assert detection is not None

    detections.append(detection)

    detections_queue.queue.put(detections)


def simulate_flight_input_worker(
    timestamp: float, odometry_queue: queue_proxy_wrapper.QueueProxyWrapper
) -> None:
    """
    Place the odometry into the queue.
    """
    # Timestamp is stored in latitude
    result, position = drone_odometry_local.DronePositionLocal.create(timestamp, 0.0, -1.0)
    assert result
    assert position is not None

    result, orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)
    assert result
    assert orientation is not None

    result, odometry = drone_odometry_local.DroneOdometryLocal.create(position, orientation)
    assert result
    assert odometry is not None

    result, odometry_time = odometry_and_time.OdometryAndTime.create(odometry)
    assert result
    assert odometry_time is not None

    odometry_time.timestamp = timestamp

    odometry_queue.queue.put(odometry_time)


def main() -> int:
    """
    Main function.
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    detections_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    odometry_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    merged_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=data_merge_worker.data_merge_worker,
        args=(
            DATA_MERGE_WORKER_TIMEOUT,
            detections_in_queue,
            odometry_in_queue,
            merged_out_queue,
            controller,
        ),
    )

    # Odometry
    for i in range(1, 10 + 1):
        simulate_flight_input_worker(i, odometry_in_queue)

    # Detection before start of odometry
    simulate_detect_target_worker(0.9, detections_in_queue)  # Discarded
    # First
    simulate_detect_target_worker(1.1, detections_in_queue)  # 1
    # Same odometry
    simulate_detect_target_worker(2.1, detections_in_queue)  # 2
    simulate_detect_target_worker(2.2, detections_in_queue)  # 2
    # Between same pair but different odometry
    simulate_detect_target_worker(3.1, detections_in_queue)  # 3
    simulate_detect_target_worker(3.9, detections_in_queue)  # 4
    # Skip odometry
    simulate_detect_target_worker(6.9, detections_in_queue)  # 7
    # Wait for odometry
    simulate_detect_target_worker(10.9, detections_in_queue)  # 11

    expected_times = [1, 2, 2, 3, 4, 7, 11]

    # Run
    worker.start()

    time.sleep(1)

    simulate_flight_input_worker(11, odometry_in_queue)
    simulate_flight_input_worker(12, odometry_in_queue)

    controller.request_exit()

    # Test
    for expected_time in expected_times:
        merged: merged_odometry_detections.MergedOdometryDetections = (
            merged_out_queue.queue.get_nowait()
        )
        assert int(merged.odometry_local.position.north) == expected_time

    assert merged_out_queue.queue.empty()

    # Teardown
    detections_in_queue.fill_and_drain_queue()
    odometry_in_queue.fill_and_drain_queue()
    worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
