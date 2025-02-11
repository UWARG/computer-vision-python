"""
Take in bounding box coordinates from Geolocation and use to estimate landing pad locations.
Returns an array of classes, each containing the x coordinate, y coordinate, and spherical 
covariance of each landing pad estimation.
"""

import os
import pathlib

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
) -> None:
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

    METHODS
    -------
    run()
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

    __decide_to_run()
        Decide when to run cluster estimation model.

    __sort_by_weights()
        Sort input model output list by weights in descending order.

    __convert_detections_to_point()
        Convert DetectionInWorld input object to a [x,y] position to store.

    __filter_by_points_ownership()
        Removes any clusters that don't have any points belonging to it.

    __filter_by_covariances()
        Removes any cluster with covariances much higher than the lowest covariance value.
    """
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
    )
    if not result:
        local_logger.error("Worker failed to create class object", True)
        return

    # Get Pylance to stop complaining
    assert estimator is not None

    while not controller.is_exit_requested():
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
