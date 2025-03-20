"""
Test worker process.
"""

import multiprocessing as mp
import pathlib
import time

import cv2
import torch

from modules.detect_target import detect_target_brightspot
from modules.detect_target import detect_target_factory
from modules.detect_target import detect_target_worker
from modules.detect_target import detect_target_ultralytics
from modules import image_and_time
from modules import detections_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


BRIGHTSPOT_TEST_PATH = pathlib.Path("tests", "brightspot_example")
IMAGE_BRIGHTSPOT_0_PATH = pathlib.Path(BRIGHTSPOT_TEST_PATH, "ir_detections_5.png")
IMAGE_BRIGHTSPOT_1_PATH = pathlib.Path(BRIGHTSPOT_TEST_PATH, "ir_detections_1.png")

BRIGHTSPOT_OPTION = detect_target_factory.DetectTargetOption.CV_BRIGHTSPOT
# Logging is identical to detect_target_ultralytics.py
# pylint: disable=duplicate-code
BRIGHTSPOT_CONFIG = detect_target_brightspot.DetectTargetBrightspotConfig(
    brightspot_percentile_threshold=99.5,
    filter_by_color=True,
    blob_color=255,
    filter_by_circularity=False,
    min_circularity=0.01,
    max_circularity=1,
    filter_by_inertia=True,
    min_inertia_ratio=0.2,
    max_inertia_ratio=1,
    filter_by_convexity=False,
    min_convexity=0.01,
    max_convexity=1,
    filter_by_area=True,
    min_area_pixels=100,
    max_area_pixels=2000,
    min_brightness_threshold=50,
    min_average_brightness_threshold=120,
)
# pylint: enable=duplicate-code

ULTRALYTICS_TEST_PATH = pathlib.Path("tests", "model_example")
IMAGE_BUS_PATH = pathlib.Path(ULTRALYTICS_TEST_PATH, "bus.jpg")
IMAGE_ZIDANE_PATH = pathlib.Path(ULTRALYTICS_TEST_PATH, "zidane.jpg")

ULTRALYTICS_OPTION = detect_target_factory.DetectTargetOption.ML_ULTRALYTICS
ULTRALYTICS_CONFIG = detect_target_ultralytics.DetectTargetUltralyticsConfig(
    0 if torch.cuda.is_available() else "cpu",
    pathlib.Path(ULTRALYTICS_TEST_PATH, "yolov8s_ultralytics_pretrained_default.pt"),
    False,
)

SHOW_ANNOTATIONS = False
SAVE_NAME = ""  # No need to save images

WORK_COUNT = 3
DELAY_FOR_SIMULATED_ULTRALYTICS_WORKER = 1  # seconds
DELAY_FOR_SIMULATED_BRIGHTSPOT_WORKER = 3  # seconds
DELAY_FOR_CUDA_WARMUP = 20  # seconds


def simulate_previous_worker(
    image_path: pathlib.Path, in_queue: queue_proxy_wrapper.QueueProxyWrapper
) -> None:
    """
    Place the image into the queue.

    image_path: Path to the image being added to the queue.
    in_queue: Input queue.
    """
    image = cv2.imread(str(image_path))  # type: ignore
    result, value = image_and_time.ImageAndTime.create(image)
    assert result
    assert value is not None
    in_queue.queue.put(value)


def run_worker(
    option: detect_target_factory.DetectTargetOption,
    config: (
        detect_target_brightspot.DetectTargetBrightspotConfig
        | detect_target_ultralytics.DetectTargetUltralyticsConfig
    ),
) -> None:
    """
    Tests target detection.

    option: Brightspot or Ultralytics.
    config: Configuration for respective target detection module.
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    image_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    image_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=detect_target_worker.detect_target_worker,
        args=(
            SAVE_NAME,
            SHOW_ANNOTATIONS,
            option,
            config,
            image_in_queue,
            image_out_queue,
            controller,
        ),
    )

    print("Starting worker")
    worker.start()

    if option == ULTRALYTICS_OPTION:
        for _ in range(0, WORK_COUNT):
            simulate_previous_worker(IMAGE_BUS_PATH, image_in_queue)

        time.sleep(DELAY_FOR_SIMULATED_ULTRALYTICS_WORKER)

        for _ in range(0, WORK_COUNT):
            simulate_previous_worker(IMAGE_ZIDANE_PATH, image_in_queue)

        # Takes some time for CUDA to warm up
        print("Waiting for CUDA to warm up")
        time.sleep(DELAY_FOR_CUDA_WARMUP)
    elif option == BRIGHTSPOT_OPTION:
        for _ in range(0, WORK_COUNT):
            simulate_previous_worker(IMAGE_BRIGHTSPOT_0_PATH, image_in_queue)

        time.sleep(DELAY_FOR_SIMULATED_BRIGHTSPOT_WORKER * 2.5)

        for _ in range(0, WORK_COUNT):
            simulate_previous_worker(IMAGE_BRIGHTSPOT_1_PATH, image_in_queue)

    controller.request_exit()
    print("Requested exit")

    # Test
    for _ in range(0, WORK_COUNT * 2):
        input_data: detections_and_time.DetectionsAndTime = image_out_queue.queue.get_nowait()
        assert input_data is not None

    assert image_out_queue.queue.empty()

    # Teardown
    print("Teardown")
    image_in_queue.fill_and_drain_queue()
    worker.join()


def main() -> int:
    """
    Main function.
    """
    run_worker(BRIGHTSPOT_OPTION, BRIGHTSPOT_CONFIG)
    # run_worker(ULTRALYTICS_OPTION, ULTRALYTICS_CONFIG)

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
