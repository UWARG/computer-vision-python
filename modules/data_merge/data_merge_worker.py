"""
Merges detections and telemetry by time.
"""

import os
import pathlib
import queue

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from .. import detections_and_time
from .. import merged_odometry_detections
from .. import odometry_and_time
from ..common.logger.modules import logger


def data_merge_worker(
    timeout: float,
    detections_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    odometry_input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process. Expects telemetry to be more frequent than detections.
    Queue is monotonic (i.e. timestamps never decrease).

    detection_input_queue, odometry_input_queue, output_queue are data queues.
    controller is how the main process communicates to this worker process.

    Merge work is done in the worker process as the queues and control mechanisms
    are naturally available.
    """
    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    assert local_logger is not None

    # Mitigate potential deadlock caused by early program exit
    try:
        previous_odometry: odometry_and_time.OdometryAndTime = odometry_input_queue.queue.get(
            timeout=timeout,
        )
        current_odometry: odometry_and_time.OdometryAndTime = odometry_input_queue.queue.get(
            timeout=timeout,
        )
    except queue.Empty:
        local_logger.error("Queue timed out on startup", True)
        return

    while not controller.is_exit_requested():
        controller.check_pause()

        detections: detections_and_time.DetectionsAndTime = detections_input_queue.queue.get()
        if detections is None:
            break

        # For initial odometry
        if detections.timestamp < previous_odometry.timestamp:
            continue

        # Advance through telemetry until detections is between previous and current
        while current_odometry.timestamp < detections.timestamp:
            previous_odometry = current_odometry
            current_odometry = odometry_input_queue.queue.get()
            if current_odometry is None:
                break

        if current_odometry is None:
            break

        # Merge with closest timestamp
        if (detections.timestamp - previous_odometry.timestamp) < (
            current_odometry.timestamp - detections.timestamp
        ):
            # Required for separation
            result, merged = merged_odometry_detections.MergedOdometryDetections.create(
                previous_odometry.odometry_data,
                detections.detections,
            )

            odometry_timestamp = previous_odometry.timestamp
        else:
            result, merged = merged_odometry_detections.MergedOdometryDetections.create(
                current_odometry.odometry_data,
                detections.detections,
            )

            odometry_timestamp = current_odometry.timestamp

        local_logger.info(
            f"Odometry timestamp: {odometry_timestamp}, detections timestamp: {detections.timestamp}, detections - odometry: {detections.timestamp - odometry_timestamp}",
            True,
        )

        if not result:
            local_logger.warning("Failed to create merged odometry and detections", True)
            continue

        local_logger.info(str(merged), True)

        # Get Pylance to stop complaining
        assert merged is not None

        output_queue.queue.put(merged)
