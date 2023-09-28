"""
Gets detections in world space and outputs estimations of objects.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import cluster_estimation


def cluster_estimation_worker(min_activation_threshold: int,
                              min_points_per_run: int,
                              random_state: int,
                              input_queue: queue_proxy_wrapper.QueueProxyWrapper,
                              output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                              worker_controller: worker_controller.WorkerController):
    """
    Worker process.

    model_path and save_name are initial settings.
    input_queue and output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    estimator_created, estimator = cluster_estimation.ClusterEstimation.create(
        min_activation_threshold,
        min_points_per_run,
        random_state)

    while not worker_controller.is_exit_requested():
        worker_controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            continue

        # TODO: When to override
        result, value = estimator.run(input_data, False)
        if not result:
            continue

        output_queue.queue.put(value)
