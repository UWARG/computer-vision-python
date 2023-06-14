"""
Tests process
TODO: PointsAndTime
"""
import multiprocessing as mp
import queue
import time

import cv2
import numpy as np

from utilities import manage_worker
from modules.detect_target import detect_target_worker
from modules import frame_and_time
# from modules import points_and_time


MODEL_PATH = "tests/model_example/yolov8s_pretrained_default.pt"
IMAGE_BUS_PATH = "tests/model_example/bus.jpg"
IMAGE_ZIDANE_PATH = "tests/model_example/zidane.jpg"

WORK_COUNT = 6


def simulate_previous_worker(image_path: str, in_queue: queue.Queue):
    """
    Place the image into the queue.
    """
    image = cv2.imread(image_path)
    data = frame_and_time.FrameAndTime(image)
    in_queue.put(data)


if __name__ == "__main__":
    # Setup
    worker_manager = manage_worker.ManageWorker()

    m = mp.Manager()
    image_in_queue = m.Queue()
    image_out_queue = m.Queue()

    worker = mp.Process(
        target=detect_target_worker.detect_target_worker,
        args=(MODEL_PATH, image_in_queue, image_out_queue, worker_manager),
    )

    # Run
    worker.start()

    for _ in range(0, int(WORK_COUNT / 2)):
        simulate_previous_worker(IMAGE_BUS_PATH, image_in_queue)

    time.sleep(1)

    for _ in range(0, int(WORK_COUNT / 2)):
        simulate_previous_worker(IMAGE_ZIDANE_PATH, image_in_queue)

    # Takes some time for CUDA to warm up
    time.sleep(20)

    worker_manager.request_exit()

    # Test
    i = 0
    while not image_out_queue.empty():
        try:
            input_data: np.ndarray = image_out_queue.get_nowait()
            assert input_data is not None
            i += 1

        except queue.Empty:
            break

    assert i == WORK_COUNT

    # Teardown
    manage_worker.ManageWorker.fill_and_drain_queue(
        image_in_queue, 1
    )
    worker.join()

    print("Done!")
