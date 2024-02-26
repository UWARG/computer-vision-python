"""
Profile detect target using full/half precision.
"""
import multiprocessing as mp
import time

import cv2
import numpy as np
import os
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
TEST_DATA_DIR = "profiler/profile_data"

THROUGHPUT_TEXT_WORK_COUNT = 50
OVERRIDE_FULL = False


def time_single_image(device: "str | int", image_path: str, use_full_precision: bool) -> float:
    detection = detect_target.DetectTarget(device, MODEL_PATH, use_full_precision)
    image = cv2.imread(image_path)
    result, value = image_and_time.ImageAndTime.create(image)

    assert result
    assert value is not None

    times = timeit.Timer(partial(detection.run, value)).repeat(10,10)
    single_time = min(times)/100
    return single_time

def time_throughput(device: "str | int", image_folder_path: str, use_full_precision: bool) -> "tuple[int, int]":
    image_names = os.listdir(image_folder_path)

    start_time = time.time_ns()
    # Setup worker
    detection = detect_target.DetectTarget(device, MODEL_PATH, use_full_precision)
    # Run
    for image_name in image_names:
        image_path = os.path.join(image_folder_path, image_name);
        image = cv2.imread(image_path)
        result, value = image_and_time.ImageAndTime.create(image)

        assert result
        assert value is not None
        status, result = detection.run(value)
    
    n_images = len(image_names)

    time_taken = time.time_ns() - start_time
    return n_images, time_taken



if __name__ == "__main__":
    # Setup
    # single image test
    device = 0 if torch.cuda.is_available() else "cpu"
    full_precision_time = time_single_image(device, IMAGE_BUS_PATH, use_full_precision = True)
    half_precision_time = time_single_image(device, IMAGE_BUS_PATH, use_full_precision = False)

    # throughput test
    n_images1, fp_worker_time = time_throughput(
        device=device,
        image_folder_path=TEST_DATA_DIR,
        use_full_precision=True
    )
    n_images2, hp_worker_time = time_throughput(
        device=device,
        image_folder_path=TEST_DATA_DIR,
        use_full_precision=False
    )
    
    # output data
    print(f"Single image full precision: {full_precision_time}")
    print(f"Single image half precision: {half_precision_time}")

    full_precision_throughput = full_precision_time / n_images1
    print(f"Full precision worker completed {n_images1} images in {full_precision_time} ns")
    print(f"Average time per image: {round(full_precision_time/n_images1)} ns")

    half_precision_throughput = half_precision_time / n_images1
    print(f"half precision worker completed {n_images1} images in {half_precision_time} ns")
    print(f"Average time per image: {round(half_precision_time/n_images1)} ns")

