"""
Test cluster_estimation_worker process.
"""

import time
import multiprocessing as mp
from typing import List

import numpy as np

from utilities.workers import queue_proxy_wrapper, worker_controller
from modules.cluster_estimation.cluster_estimation_worker import cluster_estimation_worker
from modules.detection_in_world import DetectionInWorld
from modules.object_in_world import ObjectInWorld

MIN_ACTIVATION_THRESHOLD = 3
MIN_NEW_POINTS_TO_RUN = 0
MAX_NUM_COMPONENTS = 3
RANDOM_STATE = 0
MIN_POINTS_PER_CLUSTER = 3

def check_output_results(output_queue: queue_proxy_wrapper.QueueProxyWrapper) -> None:
    """
    Checking if the output from the worker is of the correct type
    """

    while not output_queue.queue.empty():
        output_results: List[DetectionInWorld] = output_queue.queue.get()
        assert isinstance(output_results, list)
        assert all(isinstance(obj, ObjectInWorld) for obj in output_results)


def test_cluster_estimation_worker() -> int:
    """
    Integration test for cluster estimation worker.
    """

    # Worker and controller setup.
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker_process = mp.Process(
        target=cluster_estimation_worker,
        args=(
            MIN_ACTIVATION_THRESHOLD,
            MIN_NEW_POINTS_TO_RUN,
            MAX_NUM_COMPONENTS,
            RANDOM_STATE,
            MIN_POINTS_PER_CLUSTER,
            input_queue,
            output_queue,
            controller,
        ),
    )

    # Second test set: 1 clusters
    test_data_1 = [
        # Landing pad 1
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
    ]

    # First test set: 2 clusters
    test_data_2 = [
        # Landing pad 1
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        # Landing pad 2
        DetectionInWorld.create(
            np.array([[10, 10], [10, 11], [11, 11], [11, 10]]), np.array([10.5, 10.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[10.1, 10.1], [10.1, 11.1], [11.1, 11.1], [11.1, 10.1]]),
            np.array([10.6, 10.6]),
            1,
            0.92,
        )[1],
        DetectionInWorld.create(
            np.array([[9.9, 9.9], [9.9, 10.9], [10.9, 10.9], [10.9, 9.9]]),
            np.array([10.4, 10.4]),
            1,
            0.88,
        )[1],
        DetectionInWorld.create(
            np.array([[10.2, 10.2], [10.2, 11.2], [11.2, 11.2], [11.2, 10.2]]),
            np.array([10.7, 10.7]),
            1,
            0.95,
        )[1],
        DetectionInWorld.create(
            np.array([[10.3, 10.3], [10.3, 11.3], [11.3, 11.3], [11.3, 10.3]]),
            np.array([10.8, 10.8]),
            1,
            0.93,
        )[1],
    ]

    # Testing with test_data_1 (1 cluster)

    input_queue.queue.put(test_data_1)
    worker_process.start()
    time.sleep(1)

    check_output_results(output_queue)

    time.sleep(1)

    # Testing with test_data_2 (2 clusters)

    input_queue.queue.put(test_data_2)
    time.sleep(1)

    check_output_results(output_queue)

    controller.request_exit()
    input_queue.queue.put(None)
    worker_process.join()

    return 0


if __name__ == "__main__":
    result_main = test_cluster_estimation_worker()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
