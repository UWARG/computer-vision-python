"""
Merges detections and telemetry by time
"""
import queue

from utilities import manage_worker
from .. import detections_and_time
from .. import merged_odometry_detections
from .. import odometry_and_time


def data_merge_worker(detection_input_queue: queue.Queue,
                      odometry_input_queue: queue.Queue,
                      output_queue: queue.Queue,
                      worker_manager: manage_worker.ManageWorker):
    """
    Worker process. Expects telemetry to be more frequent than detections.
    Queue is monotonic (i.e. timestamps never decrease).

    detection_input_queue, odometry_input_queue, output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    # TODO: Logging?
    # Mitigate potential deadlock caused by early program exit
    try:
        previous_odometry: odometry_and_time.OdometryAndTime = odometry_input_queue.get(timeout=10)
        current_odometry: odometry_and_time.OdometryAndTime = odometry_input_queue.get(timeout=10)
    except queue.Empty:
        return

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        detections: detections_and_time.DetectionsAndTime = detection_input_queue.get()

        # For initial odometry
        if detections.timestamp < previous_odometry.timestamp:
            continue

        # Advance through telemetry until detections is between previous and current
        while current_odometry.timestamp < detections.timestamp:
            previous_odometry = current_odometry
            current_odometry = odometry_input_queue.get()

        # Merge with closest timestamp
        if detections.timestamp - previous_odometry.timestamp < current_odometry.timestamp - detections.timestamp:
            value = merged_odometry_detections.MergedOdometryDetections(
                previous_odometry.position,
                previous_odometry.orientation,
                detections.detections,
            )
        else:
            value = merged_odometry_detections.MergedOdometryDetections(
                current_odometry.position,
                current_odometry.orientation,
                detections.detections,
            )

        output_queue.put(value)
