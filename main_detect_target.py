"""
For 2022-2023 UAS competition.
"""

import argparse
import multiprocessing as mp
import pathlib

from modules.detect_target import detect_target_factory
from modules.detect_target import detect_target_worker
from modules.video_input import video_input_worker
from modules.common.logger.modules import logger
from modules.common.logger.modules import logger_setup_main
from modules.common.logger.read_yaml.modules import read_yaml
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

    # Configuration settings
    result, config = read_yaml.open_config(CONFIG_FILE_PATH)
    if not result:
        print("ERROR: Failed to load configuration file")
        return -1

    # Get Pylance to stop complaining
    assert config is not None

    # Logger configuration settings
    result, config_logger = read_yaml.open_config(logger.CONFIG_FILE_PATH)
    if not result:
        print("ERROR: Failed to load configuration file")
        return -1

    # Get Pylance to stop complaining
    assert config_logger is not None

    # Setup main logger
    result, main_logger, logging_path = logger_setup_main.setup_main_logger(config_logger)
    if not result:
        print("ERROR: Failed to create main logger")
        return -1

    # Get Pylance to stop complaining
    assert main_logger is not None
    assert logging_path is not None

    # Get settings
    try:
        # Local constants
        # pylint: disable=invalid-name
        QUEUE_MAX_SIZE = config["queue_max_size"]

        VIDEO_INPUT_CAMERA_NAME = config["video_input"]["camera_name"]
        VIDEO_INPUT_WORKER_PERIOD = config["video_input"]["worker_period"]
        VIDEO_INPUT_SAVE_NAME_PREFIX = config["video_input"]["save_prefix"]
        VIDEO_INPUT_SAVE_PREFIX = str(pathlib.Path(logging_path, VIDEO_INPUT_SAVE_NAME_PREFIX))

        DETECT_TARGET_WORKER_COUNT = config["detect_target"]["worker_count"]
        detect_target_option_int = config["detect_target"]["option"]
        DETECT_TARGET_OPTION = detect_target_factory.DetectTargetOption(detect_target_option_int)
        DETECT_TARGET_DEVICE = "cpu" if args.cpu else config["detect_target"]["device"]
        DETECT_TARGET_MODEL_PATH = config["detect_target"]["model_path"]
        DETECT_TARGET_OVERRIDE_FULL_PRECISION = args.full
        DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
        DETECT_TARGET_SAVE_PREFIX = str(pathlib.Path(logging_path, DETECT_TARGET_SAVE_NAME_PREFIX))
        DETECT_TARGET_SHOW_ANNOTATED = args.show_annotated
        # pylint: enable=invalid-name
    except KeyError as exception:
        main_logger.error(f"ERROR: Config key(s) not found: {exception}", True)
        return -1

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

    # Worker properties
    result, video_input_worker_properties = worker_manager.WorkerProperties.create(
        count=1,
        target=video_input_worker.video_input_worker,
        work_arguments=(
            VIDEO_INPUT_CAMERA_NAME,
            VIDEO_INPUT_WORKER_PERIOD,
            VIDEO_INPUT_SAVE_PREFIX,
        ),
        input_queues=[],
        output_queues=[video_input_to_detect_target_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        main_logger.error("Failed to create arguments for Video Input", True)
        return -1

    # Get Pylance to stop complaining
    assert video_input_worker_properties is not None

    result, detect_target_worker_properties = worker_manager.WorkerProperties.create(
        count=DETECT_TARGET_WORKER_COUNT,
        target=detect_target_worker.detect_target_worker,
        work_arguments=(
            DETECT_TARGET_OPTION,
            DETECT_TARGET_DEVICE,
            DETECT_TARGET_MODEL_PATH,
            DETECT_TARGET_OVERRIDE_FULL_PRECISION,
            DETECT_TARGET_SHOW_ANNOTATED,
            DETECT_TARGET_SAVE_PREFIX,
        ),
        input_queues=[video_input_to_detect_target_queue],
        output_queues=[detect_target_to_main_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        main_logger.error("Failed to create arguments for Detect Target", True)
        return -1

    # Get Pylance to stop complaining
    assert detect_target_worker_properties is not None

    # Create managers
    worker_managers = []

    result, video_input_manager = worker_manager.WorkerManager.create(
        worker_properties=video_input_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        main_logger.error("Failed to create manager for Video Input", True)
        return -1

    # Get Pylance to stop complaining
    assert video_input_manager is not None

    worker_managers.append(video_input_manager)

    result, detect_target_manager = worker_manager.WorkerManager.create(
        worker_properties=detect_target_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        main_logger.error("Failed to create manager for Detect Target", True)
        return -1

    # Get Pylance to stop complaining
    assert detect_target_manager is not None

    worker_managers.append(detect_target_manager)

    # Run
    for manager in worker_managers:
        manager.start_workers()

    while True:
        # Use main_logger for debugging
        detections_and_time = detect_target_to_main_queue.queue.get()
        if detections_and_time is None:
            break
        main_logger.debug(f"Timestamp: {detections_and_time.timestamp}", True)
        main_logger.debug(f"Num detections: {len(detections_and_time.detections)}", True)
        for detection in detections_and_time.detections:
            main_logger.debug(f"Detection: {detection}", True)

    # Teardown
    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_main_queue.fill_and_drain_queue()

    for manager in worker_managers:
        manager.join_workers()

    return 0


# pylint: enable=duplicate-code


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
