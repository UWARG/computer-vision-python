"""
Test auto-landing worker process.
"""

import datetime
import multiprocessing as mp
import pathlib
import time
import math

import numpy as np

from modules import detections_and_time
from modules import merged_odometry_detections
from modules.auto_landing import auto_landing
from modules.auto_landing import auto_landing_worker
from modules.common.modules import orientation
from modules.common.modules import position_local
from modules.common.modules.mavlink import drone_odometry_local
from modules.common.modules.logger import logger
from modules.common.modules.read_yaml import read_yaml
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


# Test-specific parameters
WORKER_PERIOD = 0.1  # seconds
LOG_TIMINGS = False  # Disable timing logging for the test
CONFIG_FILE_PATH = pathlib.Path("config.yaml")
detection_strategy = auto_landing.DetectionSelectionStrategy.FIRST_DETECTION
# detection_strategy = auto_landing.DetectionSelectionStrategy.HIGHEST_CONFIDENCE

# Ensure logs directory exists and create timestamped subdirectory
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create a timestamped subdirectory for this test session
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
test_session_dir = LOG_DIR / datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
test_session_dir.mkdir(parents=True, exist_ok=True)


def simulate_detection_input(
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    detections: list,
    position: tuple,
    orientation_angles: tuple,
) -> None:
    """
    Create and place merged odometry detections into the input queue.
    """
    # Create drone position
    result, drone_position = position_local.PositionLocal.create(
        position[0], position[1], position[2]
    )
    assert result
    assert drone_position is not None

    # Create drone orientation
    result, drone_orientation = orientation.Orientation.create(
        orientation_angles[0], orientation_angles[1], orientation_angles[2]
    )
    assert result
    assert drone_orientation is not None

    # Create drone odometry
    result, drone_odometry = drone_odometry_local.DroneOdometryLocal.create(
        drone_position, drone_orientation
    )
    assert result
    assert drone_odometry is not None

    # Create merged odometry detections
    result, merged = merged_odometry_detections.MergedOdometryDetections.create(
        drone_odometry, detections
    )
    assert result
    assert merged is not None

    input_queue.queue.put(merged)


def create_test_detection(
    bbox: list, label: int, confidence: float
) -> detections_and_time.Detection:
    """
    Create a test detection with the given parameters.
    """
    result, detection = detections_and_time.Detection.create(
        np.array(bbox, dtype=np.float32), label, confidence
    )
    assert result
    assert detection is not None
    return detection


def calculate_expected_angles(detection_bbox: list, fov_x_deg: float, fov_y_deg: float, im_w: float, im_h: float) -> tuple[float, float]:
    """
    Calculates the expected angle_x and angle_y for a given detection bounding box
    and camera parameters, mirroring the logic in auto_landing.py.
    Assumes fov_x_deg and fov_y_deg are in degrees.
    """
    x_center = (detection_bbox[0] + detection_bbox[2]) / 2
    y_center = (detection_bbox[1] + detection_bbox[3]) / 2

    angle_x = (x_center - im_w / 2) * (fov_x_deg * (math.pi / 180)) / im_w
    angle_y = (y_center - im_h / 2) * (fov_y_deg * (math.pi / 180)) / im_h
    return angle_x, angle_y


def main() -> int:
    """
    Main function.
    """
    # Logger
    test_name = pathlib.Path(__file__).stem
    result, local_logger = logger.Logger.create(test_name, LOG_TIMINGS)
    assert result  # Logger initialization should succeed
    assert local_logger is not None

    # Read config file
    result, config = read_yaml.open_config(CONFIG_FILE_PATH)
    assert result and config is not None, "Failed to read config file"

    # Check if the feature is enabled in the config
    auto_landing_config = config.get("auto_landing", {})
    if not auto_landing_config.get("enabled", False):
        local_logger.info("Auto-landing is disabled in config.yaml, skipping test.")
        return 0

    # Extract parameters from config
    try:
        fov_x = config["geolocation"]["fov_x"]
        fov_y = config["geolocation"]["fov_y"]
        image_height = config["video_input"]["height"]
        image_width = config["video_input"]["width"]
    except KeyError as e:
        local_logger.error(f"Config file missing required key: {e}")
        return -1

    

    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Create worker process using parameters from config
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            fov_x,
            fov_y,
            image_height,
            image_width,
            WORKER_PERIOD,
            detection_strategy, # Explicitly pass the strategy to be tested
            input_queue,
            output_queue,
            controller,
        ),
    )

    # Start worker
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Test 1: Send a single detection and verify it's processed
    local_logger.info("--- Test 1: Processing single detection ---")
    detection1_bbox = [200, 200, 400, 400]
    detection1 = create_test_detection(detection1_bbox, 1, 0.9)
    simulate_detection_input(
        input_queue,
        [detection1],
        (0.0, 0.0, -50.0),  # 50 meters above ground
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have output now
    assert not output_queue.queue.empty()
    landing_info = output_queue.queue.get_nowait()
    assert landing_info is not None
    assert hasattr(landing_info, "angle_x")
    assert hasattr(landing_info, "angle_y")
    assert hasattr(landing_info, "target_dist")
    local_logger.info("--- Test 1 Passed ---")

    # Test 2: Verify strategy with multiple detections
    local_logger.info(f"--- Test 2: Verifying {detection_strategy.value} strategy ---")
    
    # This is the first detection in the list
    detection_first_bbox = [100, 100, 200, 200]
    detection_first = create_test_detection(detection_first_bbox, 1, 0.7)
    
    # This detection is at the center and has higher confidence
    center_x = image_width / 2
    center_y = image_height / 2
    detection_center_bbox = [center_x, center_y, center_x, center_y]
    detection_center = create_test_detection(detection_center_bbox, 2, 0.95)

    # Order matters for FIRST_DETECTION strategy
    detections_for_test_2 = [detection_first, detection_center]

    simulate_detection_input(
        input_queue,
        detections_for_test_2,
        (0.0, 0.0, -100.0),  # 100 meters above ground
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have output
    assert not output_queue.queue.empty()
    landing_info2 = output_queue.queue.get_nowait()
    assert landing_info2 is not None

    # Determine expected angles based on the strategy
    expected_angle_x = 0.0
    expected_angle_y = 0.0

    if detection_strategy == auto_landing.DetectionSelectionStrategy.HIGHEST_CONFIDENCE:
        # If HIGHEST_CONFIDENCE, it should pick detection_center
        expected_angle_x, expected_angle_y = calculate_expected_angles(
            detection_center_bbox, fov_x, fov_y, image_width, image_height
        )
    elif detection_strategy == auto_landing.DetectionSelectionStrategy.FIRST_DETECTION:
        # If FIRST_DETECTION, it should pick detection_first
        expected_angle_x, expected_angle_y = calculate_expected_angles(
            detection_first_bbox, fov_x, fov_y, image_width, image_height
        )

    # Verify the calculated angles using a tolerance for float comparison
    assert math.isclose(landing_info2.angle_x, expected_angle_x, rel_tol=1e-7), \
        f"Expected angle_x to be {expected_angle_x}, but got {landing_info2.angle_x}"
    assert math.isclose(landing_info2.angle_y, expected_angle_y, rel_tol=1e-7), \
        f"Expected angle_y to be {expected_angle_y}, but got {landing_info2.angle_y}"
    local_logger.info("--- Test 2 Passed ---")

    # Cleanup
    controller.request_exit()

    # Drain queues
    input_queue.fill_and_drain_queue()

    # Wait for worker to finish
    worker.join(timeout=5.0)
    if worker.is_alive():
        worker.terminate()
        worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")