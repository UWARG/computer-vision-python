"""
For 2022-2023 UAS competition.
"""
import multiprocessing as mp

import cv2

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager
from modules.detect_target import detect_target_worker
from modules.video_input import video_input_worker


QUEUE_MAX_SIZE = 10

VIDEO_INPUT_CAMERA_NAME = 0
VIDEO_INPUT_WORKER_PERIOD = 1.0  # seconds
VIDEO_INPUT_SAVE_PREFIX = "log_image"

DETECT_TARGET_WORKER_COUNT = 1
DETECT_TARGET_DEVICE = 0  # Use "cpu" if no CUDA
DETECT_TARGET_MODEL_PATH = "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"  # TODO: Update
DETECT_TARGET_SAVE_PREFIX = "log_comp"


if __name__ == "__main__":
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    video_input_to_detect_target_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    detect_target_to_main_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )

    video_input_manager = worker_manager.WorkerManager()
    video_input_manager.create_workers(
        1,
        video_input_worker.video_input_worker,
        (
            VIDEO_INPUT_CAMERA_NAME,
            VIDEO_INPUT_WORKER_PERIOD,
            VIDEO_INPUT_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            controller,
        ),
    )

    detect_target_manager = worker_manager.WorkerManager()
    detect_target_manager.create_workers(
        DETECT_TARGET_WORKER_COUNT,
        detect_target_worker.detect_target_worker,
        (
            DETECT_TARGET_DEVICE,
            DETECT_TARGET_MODEL_PATH,
            DETECT_TARGET_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            detect_target_to_main_queue,
            controller,
        ),
    )

    # Run
    video_input_manager.start_workers()
    detect_target_manager.start_workers()

    while True:
        image = detect_target_to_main_queue.queue.get()
        if image is None:
            continue

        cv2.imshow("Landing Pad Detector", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Teardown
    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_main_queue.fill_and_drain_queue()

    video_input_manager.join_workers()
    detect_target_manager.join_workers()

    print("Done!")
