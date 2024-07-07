"""
For 2022-2023 UAS competition.
"""

import argparse
import datetime
import inspect
import multiprocessing as mp
import pathlib
import queue

import cv2
import yaml

from modules.detect_target import detect_target_worker
from modules.video_input import video_input_worker
from modules.logger import logger
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager


CONFIG_FILE_PATH = pathlib.Path("config.yaml")


# Code copied into main_2024.py
# pylint: disable=duplicate-code
def main() -> int:
    """
    Main function.
    """
    # Open config file
    try:
        with CONFIG_FILE_PATH.open("r", encoding="utf8") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                return -1
    except FileNotFoundError:
        print(f"File not found: {CONFIG_FILE_PATH}")
        return -1
    except IOError as exc:
        print(f"Error when opening file: {exc}")
        return -1

    # Parse whether or not to force cpu from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="option to force cpu")
    parser.add_argument("--full", action="store_true", help="option to force full precision")
    parser.add_argument(
        "--show-annotated",
        action="store_true",
        help="option to show annotated image",
    )
    args = parser.parse_args()

    # Set constants
    try:
        # Local constants
        # pylint: disable=invalid-name
        QUEUE_MAX_SIZE = config["queue_max_size"]

        LOG_DIRECTORY_PATH = config["logger"]["directory_path"]
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        VIDEO_INPUT_CAMERA_NAME = config["video_input"]["camera_name"]
        VIDEO_INPUT_WORKER_PERIOD = config["video_input"]["worker_period"]
        VIDEO_INPUT_SAVE_NAME_PREFIX = config["video_input"]["save_prefix"]
        VIDEO_INPUT_SAVE_PREFIX = (
            f"{LOG_DIRECTORY_PATH}/{start_time}/{VIDEO_INPUT_SAVE_NAME_PREFIX}"
        )

        DETECT_TARGET_WORKER_COUNT = config["detect_target"]["worker_count"]
        DETECT_TARGET_DEVICE = "cpu" if args.cpu else config["detect_target"]["device"]
        DETECT_TARGET_MODEL_PATH = config["detect_target"]["model_path"]
        DETECT_TARGET_OVERRIDE_FULL_PRECISION = args.full
        DETECT_TARGET_USE_CLASSICAL_CV = config["detect_target"]["use_classical_cv"]
        DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
        DETECT_TARGET_SAVE_PREFIX = (
            f"{LOG_DIRECTORY_PATH}/{start_time}/{DETECT_TARGET_SAVE_NAME_PREFIX}"
        )
        DETECT_TARGET_SHOW_ANNOTATED = args.show_annotated
        # pylint: enable=invalid-name
    except KeyError:
        print("Config key(s) not found")
        return -1

    pathlib.Path(LOG_DIRECTORY_PATH).mkdir(exist_ok=True)
    pathlib.Path(f"{LOG_DIRECTORY_PATH}/{start_time}").mkdir()

    result, main_logger = logger.Logger.create("main")
    if result:
        frame = inspect.currentframe()
        main_logger.info("main logger initialized", frame)

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
            DETECT_TARGET_OVERRIDE_FULL_PRECISION,
            DETECT_TARGET_USE_CLASSICAL_CV,
            DETECT_TARGET_SHOW_ANNOTATED,
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
        detections_and_time = detect_target_to_main_queue.queue.get()
        if detections_and_time is None:
            break
        print("Timestamp:" + str(detections_and_time.timestamp))
        print("Num detections:" + str(len(detections_and_time.detections)))
        for detection in detections_and_time.detections:
            print("Detection:" + str(detection))

    # Teardown
    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_main_queue.fill_and_drain_queue()

    video_input_manager.join_workers()
    detect_target_manager.join_workers()

    return 0


# pylint: enable=duplicate-code


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
