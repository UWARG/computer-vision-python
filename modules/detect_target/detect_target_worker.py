"""
Gets frames and adds a timestamp
"""
import queue

from utilities import manage_worker
from . import detect_target


def detect_target_worker(model_path: str,
                         save_name: str,
                         input_queue: queue.Queue,
                         output_queue: queue.Queue,
                         worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    model_path and save_name are initial settings.
    input_queue and output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    detector = detect_target.DetectTarget(model_path, save_name)

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        input_data = input_queue.get()
        if input_data is None:
            continue

        result, value = detector.run(input_data)
        if not result:
            continue

        output_queue.put(value)
