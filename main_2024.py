"""
For 2023-2024 UAS competition.
"""

import argparse
import inspect
import multiprocessing as mp
import pathlib
import queue

import cv2
import yaml

# Used in type annotation of flight interface output
# pylint: disable-next=unused-import
from modules import odometry_and_time
from modules.detect_target import detect_target_worker
from modules.flight_interface import flight_interface_worker
from modules.video_input import video_input_worker
from modules.data_merge import data_merge_worker
from modules.geolocation import geolocation_worker
from modules.geolocation import camera_properties
from modules.logger import logger_setup_main
from utilities import yaml
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager


CONFIG_FILE_PATH = pathlib.Path("config.yaml")


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
    result, config = yaml.open_config(CONFIG_FILE_PATH)
    if not result:
        print("ERROR: Failed to load configuration file")
        return -1

    # Get Pylance to stop complaining
    assert config is not None

    # Setup main logger
    result, main_logger, logging_path = logger_setup_main.setup_main_logger(config)
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
        DETECT_TARGET_DEVICE = "cpu" if args.cpu else config["detect_target"]["device"]
        DETECT_TARGET_MODEL_PATH = config["detect_target"]["model_path"]
        DETECT_TARGET_OVERRIDE_FULL_PRECISION = args.full
        DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
        DETECT_TARGET_SAVE_PREFIX = str(pathlib.Path(logging_path, DETECT_TARGET_SAVE_NAME_PREFIX))
        DETECT_TARGET_SHOW_ANNOTATED = args.show_annotated

        FLIGHT_INTERFACE_ADDRESS = config["flight_interface"]["address"]
        FLIGHT_INTERFACE_TIMEOUT = config["flight_interface"]["timeout"]
        FLIGHT_INTERFACE_BAUD_RATE = config["flight_interface"]["baud_rate"]
        FLIGHT_INTERFACE_WORKER_PERIOD = config["flight_interface"]["worker_period"]

        DATA_MERGE_TIMEOUT = config["data_merge"]["timeout"]

        GEOLOCATION_RESOLUTION_X = config["geolocation"]["resolution_x"]
        GEOLOCATION_RESOLUTION_Y = config["geolocation"]["resolution_y"]
        GEOLOCATION_FOV_X = config["geolocation"]["fov_x"]
        GEOLOCATION_FOV_Y = config["geolocation"]["fov_y"]
        GEOLOCATION_CAMERA_POSITION_X = config["geolocation"]["camera_position_x"]
        GEOLOCATION_CAMERA_POSITION_Y = config["geolocation"]["camera_position_y"]
        GEOLOCATION_CAMERA_POSITION_Z = config["geolocation"]["camera_position_z"]
        GEOLOCATION_CAMERA_ORIENTATION_YAW = config["geolocation"]["camera_orientation_yaw"]
        GEOLOCATION_CAMERA_ORIENTATION_PITCH = config["geolocation"]["camera_orientation_pitch"]
        GEOLOCATION_CAMERA_ORIENTATION_ROLL = config["geolocation"]["camera_orientation_roll"]
        # pylint: enable=invalid-name
    except KeyError as exception:
        frame = inspect.currentframe()
        main_logger.error(f"ERROR: Config key(s) not found: {exception}", frame)
        return -1

    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    video_input_to_detect_target_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    detect_target_to_data_merge_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    flight_interface_to_data_merge_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    # Queue size of latest odometry data must be 1
    flight_interface_to_decision_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        1,
    )
    data_merge_to_geolocation_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    geolocation_to_main_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )

    result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
        GEOLOCATION_RESOLUTION_X,
        GEOLOCATION_RESOLUTION_Y,
        GEOLOCATION_FOV_X,
        GEOLOCATION_FOV_Y,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Error creating camera intrinsics", frame)
        return -1

    result, camera_extrinsics = camera_properties.CameraDroneExtrinsics.create(
        (
            GEOLOCATION_CAMERA_POSITION_X,
            GEOLOCATION_CAMERA_POSITION_Y,
            GEOLOCATION_CAMERA_POSITION_Z,
        ),
        (
            GEOLOCATION_CAMERA_ORIENTATION_YAW,
            GEOLOCATION_CAMERA_ORIENTATION_PITCH,
            GEOLOCATION_CAMERA_ORIENTATION_ROLL,
        ),
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Error creating camera extrinsics", frame)
        return -1

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
        frame = inspect.currentframe()
        main_logger.error("Failed to create arguments for Video Input", frame)
        return -1

    # Get Pylance to stop complaining
    assert video_input_worker_properties is not None

    result, detect_target_worker_properties = worker_manager.WorkerProperties.create(
        count=DETECT_TARGET_WORKER_COUNT,
        target=detect_target_worker.detect_target_worker,
        work_arguments=(
            DETECT_TARGET_DEVICE,
            DETECT_TARGET_MODEL_PATH,
            DETECT_TARGET_OVERRIDE_FULL_PRECISION,
            DETECT_TARGET_SHOW_ANNOTATED,
            DETECT_TARGET_SAVE_PREFIX,
        ),
        input_queues=[video_input_to_detect_target_queue],
        output_queues=[detect_target_to_data_merge_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create arguments for Detect Target", frame)
        return -1

    # Get Pylance to stop complaining
    assert detect_target_worker_properties is not None

    result, flight_interface_worker_properties = worker_manager.WorkerProperties.create(
        count=1,
        target=flight_interface_worker.flight_interface_worker,
        work_arguments=(
            FLIGHT_INTERFACE_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_BAUD_RATE,
            FLIGHT_INTERFACE_WORKER_PERIOD,
        ),
        input_queues=[],
        output_queues=[flight_interface_to_data_merge_queue, flight_interface_to_decision_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create arguments for Flight Interface", frame)
        return -1

    # Get Pylance to stop complaining
    assert flight_interface_worker_properties is not None

    result, data_merge_worker_properties = worker_manager.WorkerProperties.create(
        count=1,
        target=data_merge_worker.data_merge_worker,
        work_arguments=(DATA_MERGE_TIMEOUT,),
        input_queues=[
            detect_target_to_data_merge_queue,
            flight_interface_to_data_merge_queue,
        ],
        output_queues=[data_merge_to_geolocation_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create arguments for Data Merge", frame)
        return -1

    # Get Pylance to stop complaining
    assert data_merge_worker_properties is not None

    result, geolocation_worker_properties = worker_manager.WorkerProperties.create(
        count=1,
        target=geolocation_worker.geolocation_worker,
        work_arguments=(
            camera_intrinsics,
            camera_extrinsics,
        ),
        input_queues=[data_merge_to_geolocation_queue],
        output_queues=[geolocation_to_main_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create arguments for Geolocation", frame)
        return -1

    # Get Pylance to stop complaining
    assert geolocation_worker_properties is not None

    # Create managers
    worker_managers = []

    result, video_input_manager = worker_manager.WorkerManager.create(
        worker_properties=video_input_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create manager for Video Input", frame)
        return -1

    # Get Pylance to stop complaining
    assert video_input_manager is not None

    worker_managers.append(video_input_manager)

    result, detect_target_manager = worker_manager.WorkerManager.create(
        worker_properties=detect_target_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create manager for Detect Target", frame)
        return -1

    # Get Pylance to stop complaining
    assert detect_target_manager is not None

    worker_managers.append(detect_target_manager)

    result, flight_interface_manager = worker_manager.WorkerManager.create(
        worker_properties=flight_interface_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create manager for Flight Interface", frame)
        return -1

    # Get Pylance to stop complaining
    assert flight_interface_manager is not None

    worker_managers.append(flight_interface_manager)

    result, data_merge_manager = worker_manager.WorkerManager.create(
        worker_properties=data_merge_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create manager for Data Merge", frame)
        return -1

    # Get Pylance to stop complaining
    assert data_merge_manager is not None

    worker_managers.append(data_merge_manager)

    result, geolocation_manager = worker_manager.WorkerManager.create(
        worker_properties=geolocation_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        frame = inspect.currentframe()
        main_logger.error("Failed to create manager for Geolocation", frame)
        return -1

    # Get Pylance to stop complaining
    assert geolocation_manager is not None

    worker_managers.append(geolocation_manager)

    # Run
    for manager in worker_managers:
        manager.start_workers()

    while True:
        for manager in worker_managers:
            result = manager.check_and_restart_dead_workers()
            if not result:
                frame = inspect.currentframe()
                main_logger.error("Failed to restart workers", frame)
                return -1

        try:
            geolocation_data = geolocation_to_main_queue.queue.get_nowait()
        except queue.Empty:
            geolocation_data = None

        if geolocation_data is not None:
            for detection_world in geolocation_data:
                frame = inspect.currentframe()
                main_logger.debug("Detection in world:", frame)
                main_logger.debug(
                    "geolocation vertices: " + str(detection_world.vertices.tolist()), frame
                )
                main_logger.debug(
                    "geolocation centre: " + str(detection_world.centre.tolist()), frame
                )
                main_logger.debug("geolocation label: " + str(detection_world.label), frame)
                main_logger.debug(
                    "geolocation confidence: " + str(detection_world.confidence), frame
                )

        if cv2.waitKey(1) == ord("q"):  # type: ignore
            frame = inspect.currentframe()
            main_logger.info("Exiting main loop", frame)
            break

    # Teardown
    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_data_merge_queue.fill_and_drain_queue()
    flight_interface_to_data_merge_queue.fill_and_drain_queue()
    flight_interface_to_decision_queue.fill_and_drain_queue()
    data_merge_to_geolocation_queue.fill_and_drain_queue()
    geolocation_to_main_queue.fill_and_drain_queue()

    for manager in worker_managers:
        manager.join_workers()

    cv2.destroyAllWindows()  # type: ignore

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
