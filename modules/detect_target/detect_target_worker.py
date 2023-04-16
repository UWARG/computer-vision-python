"""
Gets frames and adds a timestamp
"""

import multiprocessing as mp

from utilities import manage_worker
from . import detect_target


def detect_target_worker(model_path: str,
                         input_queue: mp.Queue, output_queue: mp.Queue,
                         worker_manager: manage_worker.ManageWorker):
    """
    Worker process.

    model_path is initial setting.
    input_queue and output_queue are data queues.
    worker_manager is how the main process communicates to this worker process.
    """
    print("detect_target_worker start")
    detector = detect_target.DetectTarget(model_path)

    while not worker_manager.is_exit_requested():
        worker_manager.check_pause()

        input_data = input_queue.get()
        print("get")
        if input_data is None:
            continue

        result, value = detector.run(input_data)
        print(result)
        if not result:
            continue

        print("put")
        output_queue.put(value)

    print("detect_target_worker done")
