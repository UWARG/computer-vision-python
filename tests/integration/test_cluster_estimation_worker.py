import time
import multiprocessing as mp
from typing import List
from queue import Queue
import numpy as np
from modules.detection_in_world import DetectionInWorld
from utilities.workers import queue_proxy_wrapper, worker_controller
from modules.cluster_estimation import cluster_estimation_worker


def test_cluster_estimation_worker() -> int:
    """
    Integration test for cluster estimation worker.
    """

    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    test_data = [
        DetectionInWorld.create(
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), np.array([1.5, 1.5]), 1, 0.9
        )[1],
        DetectionInWorld.create(
            np.array([[2, 2], [2, 3], [3, 3], [3, 2]]), np.array([2.5, 2.5]), 1, 0.85
        )[1],
    ]

    worker_process = mp.Process(
        target=cluster_estimation_worker,
        args=(
            2,
            2,
            3,
            42,
            input_queue,
            output_queue,
            controller,
        ),
    )

    worker_process.start()
    time.sleep(3)

    input_queue.queue.put(test_data)
    time.sleep(5)

    output_results: List[List[DetectionInWorld]] = []
    while not output_queue.queue.empty():
        output_results.append(output_queue.queue.get())

    print("Hello")
    print(output_results)

    # Validating Output TBD
    controller.request_exit()

    return 0


if __name__ == "__main__":
    result_main = test_cluster_estimation_worker()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")