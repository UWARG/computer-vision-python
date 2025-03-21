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
from modules.common.modules.mavlink import drone_odometry_global
from modules.common.modules import orientation
from modules.common.modules import position_global
from modules.common.modules.mavlink import local_global_conversion
from pymavlink import mavutil
import threading
import time


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

    def get_odometry(yaw, pitch, roll, lat, lon, alt, flight_mode_string) -> "tuple[bool, drone_odometry_global.DroneOdometryGlobal | None]":
        """
        Returns odometry data from the drone.
        """
        result, orientation_data = orientation.Orientation.create(
            yaw,
            pitch,
            roll,
        )
        if not result:
            return False, None

        result, position_data = position_global.PositionGlobal.create(
            lat, 
            lon, 
            alt
        )
        if not result:
            return False, None

        if flight_mode_string is None:
            flight_mode = None
            return False, None
        if flight_mode_string == "LOITER":
            flight_mode = drone_odometry_global.FlightMode.STOPPED
        if flight_mode_string == "AUTO":
            flight_mode = drone_odometry_global.FlightMode.MOVING
        else:
            flight_mode = drone_odometry_global.FlightMode.MANUAL

        # Get Pylance to stop complaining
        assert position_data is not None
        assert orientation_data is not None
        assert flight_mode is not None

        result, odometry_data = drone_odometry_global.DroneOdometryGlobal.create(
            position_data, orientation_data, flight_mode
        )
        if not result:
            return False, None

        return True, odometry_data

    def get_odometry_and_time(yaw, pitch, roll, lat, lon, alt, flight_mode_string, home_position) -> "tuple[bool, odometry_and_time.OdometryAndTime | None]":
        """
        Returns a possible OdometryAndTime with current timestamp.
        """
        result, odometry = get_odometry(yaw, pitch, roll, lat, lon, alt, flight_mode_string)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry is not None

        result, odometry_local = local_global_conversion.drone_odometry_local_from_global(
            home_position,
            odometry,
        )
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_local is not None

        result, odometry_and_time_object = odometry_and_time.OdometryAndTime.create(odometry_local)
        if not result:
            return False, None

        # Get Pylance to stop complaining
        assert odometry_and_time_object is not None

        main_logger.info(str(odometry_and_time_object), True)

        return True, odometry_and_time_object
    
    def grab_video_input(delay):
        while True:
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
            time.sleep(1/5)
    

    connection = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)

    main_logger.info("Waiting for heartbeat...")
    connection.wait_heartbeat()
    main_logger.info(f"Connected to system {connection.target_system}")

    # Request RC data at 5 Hz
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
        5,
        1
    )

    # Request position data at 5 Hz
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        5,
        1
    )

    # Request ATTITUDE data at 5 Hz
    connection.mav.request_data_stream_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # ATTITUDE is in EXTRA1
        5,
        1 
)



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

    # TODO: IS THIS NECESSARY????
    home_msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    home_lat  = home_msg.lat / 1e7  # degrees
    home_lon  = home_msg.lon / 1e7  # degrees
    home_alt  = home_msg.alt / 1000.0  # meters above MSL
    home_position = position_global.PositionGlobal.create(home_lat, home_lon, home_alt)

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


    while True:
        # TODO: choose timeout here and hz above
        msg = connection.recv_match(blocking=True, timeout=10)
        if not msg:
            continue

        msg_type = msg.get_type()

        if msg_type == 'RC_CHANNELS_RAW':
            relay = msg.chan7_raw
            main_logger.info(f"RC Channel 7: {relay}")

        elif msg_type == 'HEARTBEAT':
            # Decode mode string
            flight_mode_string = mavutil.mode_string_v10(msg)

        elif msg_type == 'GLOBAL_POSITION_INT':
            lat  = msg.lat / 1e7    # degrees
            lon  = msg.lon / 1e7    # degrees
            alt  = msg.alt / 1000.0 # meters (above MSL)
            rel_alt = msg.relative_alt / 1000.0  # meters (above home)

        elif msg_type == 'ATTITUDE':
            roll  = msg.roll   # in radians
            pitch = msg.pitch  # in radians
            yaw   = msg.yaw    # in radians
        

        if (relay == 100):
            # RELAY POSITION 0
            # THREAD (input_device to detect_target, they also had detect_target threaded to multiple as well) and (flight interface/odometry) here
            detect_target_to_data_merge_array = []
            thread = threading.Thread(target=grab_video_input)
            thread.daemon = True  # Daemon = will exit when main program exits
            thread.start()

            interface_to_data_merge_array = []
            while 1:
                msg = connection.recv_match(blocking=True, timeout=10)
                if not msg:
                    continue

                msg_type = msg.get_type()

                if msg_type == 'RC_CHANNELS_RAW':
                    relay = msg.chan7_raw
                    if relay != 100:
                        break

                elif msg_type == 'HEARTBEAT':
                    # Decode mode string
                    flight_mode_string = mavutil.mode_string_v10(msg)

                elif msg_type == 'GLOBAL_POSITION_INT':
                    lat  = msg.lat / 1e7    # degrees
                    lon  = msg.lon / 1e7    # degrees
                    alt  = msg.alt / 1000.0 # meters (above MSL)
                    rel_alt = msg.relative_alt / 1000.0  # meters (above home)

                elif msg_type == 'ATTITUDE':
                    roll  = msg.roll   # in radians
                    pitch = msg.pitch  # in radians
                    yaw   = msg.yaw    # in radians

                # time.sleep(period)

                coordinate = None # Not trying to send out any coords

                result, interface_to_data_merge = get_odometry_and_time(yaw, pitch, roll, lat, lon, alt, flight_mode_string, home_position)
                if not result:
                    continue

                interface_to_data_merge_array.append(interface_to_data_merge)
            
            thread.join()

        elif (relay > 100 and relay < 200):
            # RELAY POSITION 1
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
                connection.mav.statustext_send(
                    mavutil.mavlink.MAV_SEVERITY_INFO,  # Severity level (INFO, WARNING, ERROR, etc.)
                    str(coordinate, encoding="utf-8")         # Message must be in bytes and max 50 chars
                )
                


            for cluster_estimations in communications_to_main_array:
                if cluster_estimations is not None:
                    main_logger.debug(f"Clusters: {cluster_estimations}")
        
        # RELAY POSITION 2
        else:
            pass


    cv2.destroyAllWindows()  # type: ignore

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
