"""
Gets detections in world space and outputs estimations of objects.
"""

import os
import pathlib
import time

from modules import detection_in_world
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import cluster_estimation
from ..common.modules.logger import logger


def cluster_estimation_worker(
    min_activation_threshold: int,
    min_new_points_to_run: int,
    max_num_components: int,
    random_state: int,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
    min_points_per_cluster: int,
) -> None:
    """
    Estimation worker process.

    PARAMETERS
    ----------
    min_activation_threshold: int
        Minimum total data points before model runs.

    min_new_points_to_run: int
        Minimum number of new data points that must be collected before running model.

    max_num_components: int
        Max number of real landing pads.

    random_state: int
        Seed for randomizer, to get consistent results.

    input_queue: queue_proxy_wrapper.QueuePRoxyWrapper
        Data queue.

    output_queue: queue_proxy_wrapper.QueuePRoxyWrapper
        Data queue.

    worker_controller: worker_controller.WorkerController
        How the main process communicates to this worker process.
    """
    setup_start_time = time.time()

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    assert local_logger is not None

    local_logger.info("Logger initialized")

    result, estimator = cluster_estimation.ClusterEstimation.create(
        min_activation_threshold,
        min_new_points_to_run,
        max_num_components,
        random_state,
        local_logger,
        min_points_per_cluster,
    )
    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert estimator is not None

    setup_end_time = time.time()

    local_logger.info(
        f"{time.time()}: Worker setup took {setup_end_time - setup_start_time} seconds."
    )

    while not controller.is_exit_requested():
        iteration_start_time = time.time()

        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            local_logger.info("Recieved type None, exiting.")
            break

        is_invalid = False

        for single_input in input_data:
            if not isinstance(single_input, detection_in_world.DetectionInWorld):
                local_logger.warning(
                    f"Skipping unexpected input: {input_data}, because of unexpected value: {single_input}"
                )
                is_invalid = True
                break

        if is_invalid:
            continue

        # TODO: When to override
        result, value = estimator.run(input_data, False)
        if not result:
            continue

        output_queue.queue.put(value)

        iteration_end_time = time.time()

        local_logger.info(
            f"{time.time()}: Worker iteration took {iteration_end_time - iteration_start_time} seconds."
        )
