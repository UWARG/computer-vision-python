"""
For 2023-2024 UAS competition.
"""

import argparse
import multiprocessing as mp
import pathlib
import queue

import cv2

# Used in type annotation of flight interface output
# pylint: disable-next=unused-import
from modules import odometry_and_time
from modules.common.modules.camera import camera_factory
from modules.common.modules.camera import camera_opencv
from modules.common.modules.camera import camera_picamera2
from modules.communications import communications_worker
from modules.detect_target import detect_target_brightspot
from modules.detect_target import detect_target_factory
from modules.detect_target import detect_target_worker
from modules.detect_target import detect_target_ultralytics
from modules.flight_interface import flight_interface_worker
from modules.video_input import video_input_worker
from modules.data_merge import data_merge_worker
from modules.geolocation import geolocation_worker
from modules.geolocation import camera_properties
from modules.cluster_estimation import cluster_estimation_worker
from modules.common.modules.logger import logger
from modules.common.modules.logger import logger_main_setup
from modules.common.modules.read_yaml import read_yaml
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager

from modules.video_input import video_input
from modules.flight_interface import flight_interface
from modules import detections_and_time
from modules import merged_odometry_detections
from modules import odometry_and_time
from modules.geolocation import geolocation
from modules.cluster_estimation import cluster_estimation
from modules import detection_in_world
from modules.communications import communications
from modules import object_in_world


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
    result, main_logger, logging_path = logger_main_setup.setup_main_logger(config_logger)
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

        VIDEO_INPUT_WORKER_PERIOD = config["video_input"]["worker_period"]
        VIDEO_INPUT_OPTION = camera_factory.CameraOption(config["video_input"]["camera_enum"])
        VIDEO_INPUT_WIDTH = config["video_input"]["width"]
        VIDEO_INPUT_HEIGHT = config["video_input"]["height"]
        match VIDEO_INPUT_OPTION:
            case camera_factory.CameraOption.OPENCV:
                VIDEO_INPUT_CAMERA_CONFIG = camera_opencv.ConfigOpenCV(
                    **config["video_input"]["camera_config"]
                )
            case camera_factory.CameraOption.PICAM2:
                VIDEO_INPUT_CAMERA_CONFIG = camera_picamera2.ConfigPiCamera2(
                    **config["video_input"]["camera_config"]
                )
            case _:
                main_logger.error(f"Inputted an invalid camera option: {VIDEO_INPUT_OPTION}", True)
                return -1

        VIDEO_INPUT_IMAGE_NAME = (
            config["video_input"]["image_name"] if config["video_input"]["log_images"] else None
        )

        DETECT_TARGET_WORKER_COUNT = config["detect_target"]["worker_count"]
        DETECT_TARGET_OPTION = detect_target_factory.DetectTargetOption(
            config["detect_target"]["option"]
        )
        DETECT_TARGET_SAVE_PREFIX = str(
            pathlib.Path(logging_path, config["detect_target"]["save_prefix"])
        )
        DETECT_TARGET_SHOW_ANNOTATED = args.show_annotated
        match DETECT_TARGET_OPTION:
            case detect_target_factory.DetectTargetOption.ML_ULTRALYTICS:
                DETECT_TARGET_CONFIG = detect_target_ultralytics.DetectTargetUltralyticsConfig(
                    config["detect_target"]["config"]["device"],
                    config["detect_target"]["config"]["model_path"],
                    args.full,
                )
            case detect_target_factory.DetectTargetOption.CV_BRIGHTSPOT:
                DETECT_TARGET_CONFIG = detect_target_brightspot.DetectTargetBrightspotConfig(
                    **config["detect_target"]["config"]
                )
            case _:
                main.logger.error(
                    f"Inputted an invalid detect target option: {DETECT_TARGET_OPTION}", True
                )
                return -1

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

        MIN_ACTIVATION_THRESHOLD = config["cluster_estimation"]["min_activation_threshold"]
        MIN_NEW_POINTS_TO_RUN = config["cluster_estimation"]["min_new_points_to_run"]
        MAX_NUM_COMPONENTS = config["cluster_estimation"]["max_num_components"]
        RANDOM_STATE = config["cluster_estimation"]["random_state"]

        COMMUNICATIONS_TIMEOUT = config["communications"]["timeout"]
        COMMUNICATIONS_WORKER_PERIOD = config["communications"]["worker_period"]

        # pylint: enable=invalid-name
    except KeyError as exception:
        main_logger.error(f"Config key(s) not found: {exception}", True)
        return -1
    except ValueError as exception:
        main_logger.error(f"{exception}", True)
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
    flight_interface_to_communications_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    data_merge_to_geolocation_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    geolocation_to_cluster_estimation_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    flight_interface_decision_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    cluster_estimation_to_communications_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    communications_to_flight_interface_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    communications_to_main_queue = queue_proxy_wrapper.QueueProxyWrapper(
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
        main_logger.error("Error creating camera intrinsics", True)
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
        main_logger.error("Error creating camera extrinsics", True)
        return -1



    # From here on, we cook


    result, input_device = video_input.VideoInput.create(
        VIDEO_INPUT_OPTION,
        VIDEO_INPUT_WIDTH,
        VIDEO_INPUT_HEIGHT,
        VIDEO_INPUT_CAMERA_CONFIG,
        VIDEO_INPUT_IMAGE_NAME,
        main_logger
    )
    if not result:
        main_logger.error("Worker failed to create class object")
        return -1

    # Get Pylance to stop complaining
    assert input_device is not None

    result, detector = detect_target_factory.create_detect_target(
        DETECT_TARGET_SAVE_PREFIX,
        DETECT_TARGET_SHOW_ANNOTATED,
        DETECT_TARGET_OPTION,
        DETECT_TARGET_CONFIG,
        main_logger
    )
    if not result:
        main_logger.error("Could not construct detector.")
        return -1

    # Get Pylance to stop complaining
    assert detector is not None

    result, interface = flight_interface.FlightInterface.create(
        FLIGHT_INTERFACE_ADDRESS,
        FLIGHT_INTERFACE_TIMEOUT,
        FLIGHT_INTERFACE_BAUD_RATE,
        main_logger
    )
    if not result:
        main_logger.error("Worker failed to create class object", True)
        return -1

    # Get Pylance to stop complaining
    assert interface is not None

    # TODO: IS THIS NECESSARY????
    home_position = interface.get_home_position()

    result, locator = geolocation.Geolocation.create(
        camera_intrinsics,
        camera_extrinsics,
        main_logger,
    )
    if not result:
        main_logger.error("Worker failed to create class object")
        return -1

    # Get Pylance to stop complaining
    assert locator is not None


    result, estimator = cluster_estimation.ClusterEstimation.create(
        MIN_ACTIVATION_THRESHOLD,
        MIN_NEW_POINTS_TO_RUN,
        MAX_NUM_COMPONENTS,
        RANDOM_STATE,
        main_logger
    )
    if not result:
        main_logger.error("Worker failed to create class object", True)
        return -1

    # Get Pylance to stop complaining
    assert estimator is not None

    main_logger.info(f"Home position received: {home_position}", True)

    result, comm = communications.Communications.create(home_position, main_logger)
    if not result:
        main_logger.error("Worker failed to create class object", True)
        return -1

    # Get Pylance to stop complaining
    assert comm is not None



    # RELAY POSITION 0
    # THREAD (input_device to detect_target, they also had detect_target threaded to multiple as well) and (flight interface/odometry) here
    detect_target_to_data_merge_array = []
    while 1:

        result, input_device_to_detector = input_device.run()
        if not result:
            continue

        #grab from input_device and put into detect_target here
        input_data = input_device_to_detector
        if input_data is None:
            main_logger.info("Recieved type None, exiting.")
            break

        result, detect_target_to_data_merge = detector.run(input_data)
        if not result:
            continue

        detect_target_to_data_merge_array.append(detect_target_to_data_merge)


    interface_to_data_merge_array = []
    while 1:

        # time.sleep(period)

        coordinate = None # Not trying to send out any coords

        result, interface_to_data_merge = interface.run(coordinate)
        if not result:
            continue

        interface_to_data_merge_array.append(interface_to_data_merge)


    # RELAY POSITION 1
    while 1: #while 1 but realistically we just want it to do it once so not really...
        data_merge_to_geolocation_array = []
        i = 0
        previous_odometry = interface_to_data_merge_array[i]
        i += 1
        current_odometry = interface_to_data_merge_array[i]
        for detections in detect_target_to_data_merge_array:
            detections: detections_and_time.DetectionsAndTime

            # For initial odometry
            if detections.timestamp < previous_odometry.timestamp:
                continue

            while current_odometry.timestamp < detections.timestamp:
                previous_odometry = current_odometry
                i += 1
                if len(interface_to_data_merge_array) == i:
                    break
                current_odometry = interface_to_data_merge_array[i]
            
            if len(interface_to_data_merge_array) == i:
                break

            # Merge with closest timestamp
            if (detections.timestamp - previous_odometry.timestamp) < (
                current_odometry.timestamp - detections.timestamp
            ):
                # Required for separation
                result, merged = merged_odometry_detections.MergedOdometryDetections.create(
                    previous_odometry.odometry_data,
                    detections.detections,
                )

                odometry_timestamp = previous_odometry.timestamp
            else:
                result, merged = merged_odometry_detections.MergedOdometryDetections.create(
                    current_odometry.odometry_data,
                    detections.detections,
                )

                odometry_timestamp = current_odometry.timestamp

            main_logger.info(
                f"Odometry timestamp: {odometry_timestamp}, detections timestamp: {detections.timestamp}, detections - odometry: {detections.timestamp - odometry_timestamp}",
                True,
            )

            if not result:
                main_logger.warning("Failed to create merged odometry and detections", True)
                continue

            main_logger.info(str(merged), True)

            # Get Pylance to stop complaining
            assert merged is not None

            data_merge_to_geolocation_array.append(merged)


        geolocation_to_cluster_estimation_array = []
        for input_data in data_merge_to_geolocation_array:
            if not isinstance(input_data, merged_odometry_detections.MergedOdometryDetections):
                main_logger.warning(f"Skipping unexpected input: {input_data}")
                continue
            result, geolocation_to_cluster_estimation = locator.run(input_data)
            if not result:
                continue

            geolocation_to_cluster_estimation_array.append(geolocation_to_cluster_estimation)


        cluster_estimation_to_communications_array = []
        for input_data in geolocation_to_cluster_estimation_array:

            is_invalid = False

            for single_input in input_data:
                if not isinstance(single_input, detection_in_world.DetectionInWorld):
                    main_logger.warning(
                        f"Skipping unexpected input: {input_data}, because of unexpected value: {single_input}"
                    )
                    is_invalid = True
                    break

            if is_invalid:
                continue

            # TODO: When to override
            result, cluster_estimation_to_communications = estimator.run(input_data, False)
            if not result:
                continue

            cluster_estimation_to_communications_array.append(cluster_estimation_to_communications)

        
        communications_to_main_array = []
        communications_to_flight_interface_array = []
        for input_data in cluster_estimation_to_communications_array:
            is_invalid = False

            for single_input in input_data:
                if not isinstance(single_input, object_in_world.ObjectInWorld):
                    main_logger.warning(
                        f"Skipping unexpected input: {input_data}, because of unexpected value: {single_input}"
                    )
                    is_invalid = True
                    break

            if is_invalid:
                continue

            result, metadata, list_of_messages = comm.run(input_data)
            if not result:
                continue

            communications_to_main_array.append(metadata)
            communications_to_flight_interface_array.append(metadata)

            for message in list_of_messages:

                # time.sleep(period)

                communications_to_main_array.append(message)
                communications_to_flight_interface_array.append(message)
        
        for coordinate in communications_to_flight_interface_array:
            result, value = interface.run(coordinate)
            if not result:
                continue
            
            #ignore value here


        for cluster_estimations in communications_to_main_array:
            if cluster_estimations is not None:
                main_logger.debug(f"Clusters: {cluster_estimations}")


    # RELAY POSITION 2
    while 1:
        # chill
        pass

    cv2.destroyAllWindows()  # type: ignore

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
