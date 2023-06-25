"""
For 2022-2023 UAS competition.
"""
import multiprocessing as mp

import cv2
import torch

from utilities import manage_worker
from modules.detect_target import detect_target_worker
from modules.video_input import video_input_worker


QUEUE_MAX_SIZE = 10

VIDEO_INPUT_CAMERA_NAME = 0
VIDEO_INPUT_WORKER_PERIOD = 1.0  # seconds

DETECT_TARGET_WORKER_COUNT = 1
DETECT_TARGET_MODEL_PATH = "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"  # TODO: Update
DETECT_TARGET_SAVE_PREFIX = "log_comp"


def create_workers(count: int, target, args: "tuple") -> "list[mp.Process]":
    """
    Create a list of identical worker processes.
    """
    workers = []
    for _ in range(0, count):
        worker = mp.Process(target=target, args=args)
        workers.append(worker)

    return workers


def start_workers(workers: "list[mp.Process]"):
    """
    Start worker processes.
    """
    for worker in workers:
        worker.start()


def join_workers(workers: "list[mp.Process]"):
    """
    Join worker processes.
    """
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    # CUDA check, comment out if testing on a computer without CUDA
    assert torch.cuda.is_available()

    # Setup
    worker_manager = manage_worker.ManageWorker()

    m = mp.Manager()
    video_input_to_detect_target_queue = m.Queue(QUEUE_MAX_SIZE)
    detect_target_to_main_queue = m.Queue(QUEUE_MAX_SIZE)

    video_input_workers = create_workers(
        1,
        video_input_worker.video_input_worker,
        (
            VIDEO_INPUT_CAMERA_NAME,
            VIDEO_INPUT_WORKER_PERIOD,
            video_input_to_detect_target_queue,
            worker_manager,
        ),
    )

    detect_target_workers = create_workers(
        DETECT_TARGET_WORKER_COUNT,
        detect_target_worker.detect_target_worker,
        (
            DETECT_TARGET_MODEL_PATH,
            DETECT_TARGET_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            detect_target_to_main_queue,
            worker_manager,
        ),
    )

    # Run
    start_workers(video_input_workers)
    start_workers(detect_target_workers)

    while True:
        image = detect_target_to_main_queue.get()
        if image is None:
            continue

        cv2.imshow("Landing Pad Detector", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Teardown
    worker_manager.request_exit()

    manage_worker.ManageWorker.fill_and_drain_queue(
        video_input_to_detect_target_queue,
        QUEUE_MAX_SIZE,
    )

    manage_worker.ManageWorker.fill_and_drain_queue(
        detect_target_to_main_queue,
        QUEUE_MAX_SIZE,
    )

    join_workers(video_input_workers)
    join_workers(detect_target_workers)

    print("Done!")
