"""
Test worker process.
"""

import multiprocessing as mp
import pathlib
import time

import cv2
import torch

from modules.detect_target import detect_target_worker
from modules import image_and_time
from modules import detections_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


TEST_PATH = pathlib.Path("tests", "model_example")
IMAGE_BUS_PATH = pathlib.Path(TEST_PATH, "bus.jpg")
IMAGE_ZIDANE_PATH = pathlib.Path(TEST_PATH, "zidane.jpg")

WORK_COUNT = 3
DELAY_FOR_SIMULATED_WORKERS = 1  # seconds
DELAY_FOR_CUDA_WARMUP = 20  # seconds

MODEL_PATH = pathlib.Path(TEST_PATH, "yolov8s_ultralytics_pretrained_default.pt")
OVERRIDE_FULL = False
USE_CLASSICAL_CV = False
SHOW_ANNOTATIONS = False
SAVE_NAME = ""  # No need to save images


def simulate_previous_worker(
    image_path: pathlib.Path, in_queue: queue_proxy_wrapper.QueueProxyWrapper
) -> None:
    """
    Place the image into the queue.
    """
    image = cv2.imread(str(image_path))  # type: ignore
    result, value = image_and_time.ImageAndTime.create(image)
    assert result
    assert value is not None
    in_queue.queue.put(value)


def main() -> int:
    """
    Main function.
    """
    # Setup
    # Not a constant
    # pylint: disable-next=invalid-name
    device = 0 if torch.cuda.is_available() else "cpu"
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    image_in_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    image_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=detect_target_worker.detect_target_worker,
        args=(
            device,
            MODEL_PATH,
            OVERRIDE_FULL,
            USE_CLASSICAL_CV,
            SHOW_ANNOTATIONS,
            SAVE_NAME,
            image_in_queue,
            image_out_queue,
            controller,
        ),
    )

    # Run
    print("Starting worker")
    worker.start()

    for _ in range(0, WORK_COUNT):
        simulate_previous_worker(IMAGE_BUS_PATH, image_in_queue)

    time.sleep(DELAY_FOR_SIMULATED_WORKERS)

    for _ in range(0, WORK_COUNT):
        simulate_previous_worker(IMAGE_ZIDANE_PATH, image_in_queue)

    # Takes some time for CUDA to warm up
    print("Waiting for CUDA to warm up")
    time.sleep(DELAY_FOR_CUDA_WARMUP)

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

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
