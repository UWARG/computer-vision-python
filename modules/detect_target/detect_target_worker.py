"""
Gets frames and outputs detections in image space.
"""

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import detect_target


# Extra parameters required for worker communication
# pylint: disable=too-many-arguments
def detect_target_worker(device: "str | int",
                         model_path: str,
                         override_full: bool,
                         save_name: str,
                         show_annotations: bool,
                         input_queue: queue_proxy_wrapper.QueueProxyWrapper,
                         output_queue: queue_proxy_wrapper.QueueProxyWrapper,
                         controller: worker_controller.WorkerController):
    """
    Worker process.

    device, model_path, override_full, and save_name are initial settings.
    input_queue and output_queue are data queues.
    controller is how the main process communicates to this worker process.
    """
    detector = detect_target.DetectTarget(device, model_path, override_full, show_annotations, save_name)

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            break

        result, value = detector.run(input_data)
        if not result:
            continue

        output_queue.queue.put(value)

# pylint: enable=too-many-arguments
