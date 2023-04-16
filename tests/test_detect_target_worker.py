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


MODEL_PATH = "tests/model_example/yolov8s.pt"
IMAGE_BUS_PATH = "tests/model_example/bus.jpg"
IMAGE_ZIDANE_PATH = "tests/model_example/zidane.jpg"


def simulate_previous_worker(image_path: str, in_queue: mp.Queue):
    image = cv2.imread(image_path)
    data = frame_and_time.FrameAndTime(image)
    in_queue.put(data)


if __name__ == "__main__":
    # Setup
    worker_manager = manage_worker.ManageWorker()
    image_in_queue = mp.Queue()
    image_out_queue = mp.Queue()

    worker = mp.Process(
        target=detect_target_worker.detect_target_worker,
        args=(MODEL_PATH, image_in_queue, image_out_queue, worker_manager)
    )

    # Run
    worker.start()

    for _ in range(0, 3):
        simulate_previous_worker(IMAGE_BUS_PATH, image_in_queue)

    time.sleep(1)

    for _ in range(0, 3):
        simulate_previous_worker(IMAGE_ZIDANE_PATH, image_in_queue)


    time.sleep(30)

    worker_manager.request_exit()

    print(image_out_queue.qsize())

    # Test
    i = 0
    while not image_out_queue.empty():
        try:
            input_data: np.ndarray = image_out_queue.get_nowait()
            assert input_data is not None
            i += 1
            print(input_data.shape)
            time.sleep(0.01)

        except queue.Empty:
            break

    print(i)
    assert i == 6

    print("Done!")
