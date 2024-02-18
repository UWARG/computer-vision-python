"""
Gets detections in world space and outputs estimations of objects.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import cluster_estimation


def cluster_estimation_worker(
    min_activation_threshold: int,
    min_new_points_to_run: int,
    random_state: int,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
):
    """
    Estimation worker process.

    PARAMETERS
    ----------
    min_activation_threshold: int
        Minimum total data points before model runs.

    min_new_points_to_run: int
        Minimum number of new data points that must be collected before running model.

    random_state: int
        Seed for randomizer, to get consistent results.

    input_queue: queue_proxy_wrapper.QueuePRoxyWrapper
        Data queue.

    output_queue: queue_proxy_wrapper.QueuePRoxyWrapper
        Data queue.

    worker_controller: worker_controller.WorkerController
        How the main process communicates to this worker process.
    """
    result, estimator = cluster_estimation.ClusterEstimation.create(
        min_activation_threshold,
        min_new_points_to_run,
        random_state,
    )
    if not result:
        print("ERROR: Worker failed to create class object")
        return

    # Get Pylance to stop complaining
    assert estimator is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            continue

        # TODO: When to override
        result, value = estimator.run(input_data, False)
        if not result:
            continue

        output_queue.queue.put(value)
