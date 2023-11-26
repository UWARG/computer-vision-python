"""
Profile detect target using full/half precision.
"""
import multiprocessing as mp
import time

import cv2
import numpy as np
import timeit
import torch

from functools import partial
from modules.detect_target import detect_target, detect_target_worker
from modules import image_and_time
# from modules import points_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


MODEL_PATH = "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"
IMAGE_BUS_PATH = "tests/model_example/bus.jpg"
IMAGE_ZIDANE_PATH = "tests/model_example/zidane.jpg"

THROUGHPUT_TEXT_WORK_COUNT = 50
OVERRIDE_FULL = False


def test_single_image(image_path: str, use_full_precision: bool) -> float:
    device = 0 if torch.cuda.is_available() else "cpu"
    detection = detect_target.DetectTarget(device, MODEL_PATH, use_full_precision)
    image = cv2.imread(image_path)
    result, value = image_and_time.ImageAndTime.create(image)

    assert result
    assert value is not None

    times = timeit.Timer(partial(detection.run, value)).repeat(5,5)
    single_time = min(times)/100
    return single_time


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
    # single image test
    full_precision_time = test_single_image(IMAGE_BUS_PATH, use_full_precision = True)
    half_precision_time = test_single_image(IMAGE_BUS_PATH, use_full_precision = False)
    print(f"Single image full precision: {full_precision_time}")
    print(f"Single image half precision: {half_precision_time}")

    # throughput test

