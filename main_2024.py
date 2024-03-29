"""
For 2023-2024 UAS competition.
"""

import argparse
import datetime
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
from modules.logger import logger
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager


CONFIG_FILE_PATH = pathlib.Path("config.yaml")


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

        LOG_DIRECTORY_PATH = config["log_directory_path"]
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

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
        DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
        DETECT_TARGET_SAVE_PREFIX = (
            f"{LOG_DIRECTORY_PATH}/{start_time}/{DETECT_TARGET_SAVE_NAME_PREFIX}"
        )
        DETECT_TARGET_SHOW_ANNOTATED = args.show_annotated

        FLIGHT_INTERFACE_ADDRESS = config["flight_interface"]["address"]
        FLIGHT_INTERFACE_TIMEOUT = config["flight_interface"]["timeout"]
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
    except KeyError:
        print("Config key(s) not found")
        return -1

    pathlib.Path(LOG_DIRECTORY_PATH).mkdir(exist_ok=True)
    pathlib.Path(f"{LOG_DIRECTORY_PATH}/{start_time}").mkdir()

    result, main_logger = logger.Logger.create("main")
    if result:
        main_logger.logger.info("main logger initialized")

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
    data_merge_to_geolocation_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    geolocation_to_main_queue = queue_proxy_wrapper.QueueProxyWrapper(
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
            DETECT_TARGET_SHOW_ANNOTATED,
            DETECT_TARGET_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            detect_target_to_data_merge_queue,
            controller,
        ),
    )

    flight_interface_manager = worker_manager.WorkerManager()
    flight_interface_manager.create_workers(
        1,
        flight_interface_worker.flight_interface_worker,
        (
            FLIGHT_INTERFACE_ADDRESS,
            FLIGHT_INTERFACE_TIMEOUT,
            FLIGHT_INTERFACE_WORKER_PERIOD,
            flight_interface_to_data_merge_queue,
            controller,
        ),
    )

    data_merge_manager = worker_manager.WorkerManager()
    data_merge_manager.create_workers(
        1,
        data_merge_worker.data_merge_worker,
        (
            DATA_MERGE_TIMEOUT,
            detect_target_to_data_merge_queue,
            flight_interface_to_data_merge_queue,
            data_merge_to_geolocation_queue,
            controller,
        ),
    )

    result, camera_intrinsics = camera_properties.CameraIntrinsics.create(
        GEOLOCATION_RESOLUTION_X,
        GEOLOCATION_RESOLUTION_Y,
        GEOLOCATION_FOV_X,
        GEOLOCATION_FOV_Y,
    )
    if not result:
        print("Error creating camera intrinsics")
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
        print("Error creating camera extrinsics")
        return -1

    geolocation_manager = worker_manager.WorkerManager()
    geolocation_manager.create_workers(
        1,
        geolocation_worker.geolocation_worker,
        (
            camera_intrinsics,
            camera_extrinsics,
            data_merge_to_geolocation_queue,
            geolocation_to_main_queue,
            controller,
        ),
    )

    # Run
    video_input_manager.start_workers()
    detect_target_manager.start_workers()
    flight_interface_manager.start_workers()
    data_merge_manager.start_workers()
    geolocation_manager.start_workers()

    while True:
        try:
            geolocation_data = geolocation_to_main_queue.queue.get_nowait()
        except queue.Empty:
            geolocation_data = None

        if geolocation_data is not None:
            for detection_world in geolocation_data:
                print("geolocation vertices: " + str(detection_world.vertices.tolist()))
                print("geolocation centre: " + str(detection_world.centre.tolist()))
                print("geolocation label: " + str(detection_world.label))
                print("geolocation confidence: " + str(detection_world.confidence))
                print("")

        if cv2.waitKey(1) == ord("q"):  # type: ignore
            print("Exiting main loop")
            break

    # Teardown
    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_data_merge_queue.fill_and_drain_queue()
    flight_interface_to_data_merge_queue.fill_and_drain_queue()
    data_merge_to_geolocation_queue.fill_and_drain_queue()
    geolocation_to_main_queue.fill_and_drain_queue()

    video_input_manager.join_workers()
    detect_target_manager.join_workers()
    flight_interface_manager.join_workers()
    data_merge_manager.join_workers()
    geolocation_manager.join_workers()

    cv2.destroyAllWindows()  # type: ignore

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
