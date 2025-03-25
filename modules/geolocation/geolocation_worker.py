"""
Convert bounding box data into ground data.
"""

import os
import pathlib
import time

from modules import merged_odometry_detections
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import camera_properties
from . import geolocation
from ..common.modules.logger import logger


def geolocation_worker(
    camera_intrinsics: camera_properties.CameraIntrinsics,
    camera_drone_extrinsics: camera_properties.CameraDroneExtrinsics,
    log_timings: bool,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    input_queue and output_queue are data queues.
    controller is how the main process communicates to this worker process.
    """
    # TODO: Handle errors better
    setup_start_time = time.time() if log_timings else None

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    assert local_logger is not None

    local_logger.info("Logger initialized")

    result, locator = geolocation.Geolocation.create(
        camera_intrinsics,
        camera_drone_extrinsics,
        local_logger,
    )
    if not result:
        local_logger.error("Worker failed to create class object")
        return

    # Get Pylance to stop complaining
    assert locator is not None

    if log_timings:
        setup_end_time = time.time()
        local_logger.info(
            f"{time.time()}: Worker setup took {setup_end_time - setup_start_time} seconds."
        )

    while not controller.is_exit_requested():
        iteration_start_time = time.time() if log_timings else None

        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            local_logger.info("Recieved type None, exiting.")
            break

        if not isinstance(input_data, merged_odometry_detections.MergedOdometryDetections):
            local_logger.warning(f"Skipping unexpected input: {input_data}")
            continue

        result, value = locator.run(input_data)
        if not result:
            continue

        output_queue.queue.put(value)

        if log_timings:
            iteration_end_time = time.time()
            local_logger.info(
                f"{time.time()}: Worker iteration took {iteration_end_time - iteration_start_time} seconds."
            )
