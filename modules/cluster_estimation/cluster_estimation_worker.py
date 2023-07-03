"""
Gets detections in world space and outputs estimations of objects
"""
import queue

from utilities import manage_worker
from . import cluster_estimation


def cluster_estimation_worker(# TODO: Initialization variables
                              input_queue: queue.Queue,
                              output_queue: queue.Queue,
                              worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    model_path and save_name are initial settings.
    input_queue and output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    estimator = cluster_estimation.ClusterEstimation()  # TODO: Initialization variables

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        input_data = input_queue.get()
        if input_data is None:
            continue

        # TODO: When to override
        result, value = estimator.run(input_data, False)
        if not result:
            continue

        output_queue.put(value)
