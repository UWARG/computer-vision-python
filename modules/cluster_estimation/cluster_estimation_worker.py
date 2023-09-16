"""
Gets detections in world space and outputs estimations of objects
"""
from queue import Queue

from utilities import manage_worker
from . import cluster_estimation


def cluster_estimation_worker(# TODO: Initialization variables
                              input_queue: Queue,
                              output_queue: Queue,
                              worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    model_path and save_name are initial settings.
    input_queue and output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    MIN_ACTIVATION_THRESHOLD = 100
    MIN_POINTS_PER_RUN = 10
    RANDOM_STATE = 0

    estimator_created, estimator = cluster_estimation.ClusterEstimation.create(MIN_ACTIVATION_THRESHOLD,
                                                                               MIN_POINTS_PER_RUN,
                                                                               RANDOM_STATE) 

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
