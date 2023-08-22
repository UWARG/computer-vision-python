"""
Test worker process.
TODO: PointsAndTime
"""
import multiprocessing as mp
import time

import cv2
import numpy as np
import torch

from modules.detect_target import detect_target_worker
from modules import image_and_time
# from modules import points_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


MODEL_PATH = "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"
IMAGE_BUS_PATH = "tests/model_example/bus.jpg"
IMAGE_ZIDANE_PATH = "tests/model_example/zidane.jpg"

WORK_COUNT = 3


def simulate_previous_worker(image_path: str, in_queue: queue_proxy_wrapper.QueueProxyWrapper):
    """
    Place the image into the queue.
    """
    image = cv2.imread(image_path)
    result, value = image_and_time.ImageAndTime.create(image)
    assert result
    assert value is not None
    in_queue.queue.put(value)


if __name__ == "__main__":
    # Setup
    device = 0 if torch.cuda.is_available() else "cpu"
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    image_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    image_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=detect_target_worker.detect_target_worker,
        args=(device, MODEL_PATH, False, "", image_in_queue, image_out_queue, controller),
    )

    # Run
    worker.start()

    for _ in range(0, WORK_COUNT):
        simulate_previous_worker(IMAGE_BUS_PATH, image_in_queue)

    time.sleep(1)

    for _ in range(0, WORK_COUNT):
        simulate_previous_worker(IMAGE_ZIDANE_PATH, image_in_queue)

    # Takes some time for CUDA to warm up
    time.sleep(20)

    controller.request_exit()

    # Test
    for _ in range(0, WORK_COUNT * 2):
        input_data: np.ndarray = image_out_queue.queue.get_nowait()
        assert input_data is not None

    assert image_out_queue.queue.empty()

    # Teardown
    image_in_queue.fill_and_drain_queue()
    worker.join()

    print("Done!")
