"""
Test auto-landing worker process.
"""

import datetime
import multiprocessing as mp
import pathlib
import time

import numpy as np

from modules import detections_and_time
from modules import merged_odometry_detections
from modules.auto_landing import auto_landing
from modules.auto_landing import auto_landing_worker
from modules.common.modules import orientation
from modules.common.modules import position_local
from modules.common.modules.mavlink import drone_odometry_local
from modules.common.modules.logger import logger
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller

# Worker parameters
FOV_X = 90.0  # degrees
FOV_Y = 90.0  # degrees
IMAGE_HEIGHT = 640.0  # pixels
IMAGE_WIDTH = 640.0  # pixels
WORKER_PERIOD = 0.1  # seconds
# The worker now defaults to HIGHEST_CONFIDENCE, so we test for that
DETECTION_STRATEGY = auto_landing.DetectionSelectionStrategy.HIGHEST_CONFIDENCE
LOG_TIMINGS = False  # Disable timing logging for the test

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


def main() -> int:
    """
    Main function.
    """
    # Logger
    test_name = pathlib.Path(__file__).stem
    result, local_logger = logger.Logger.create(test_name, LOG_TIMINGS)
    assert result  # Logger initialization should succeed
    assert local_logger is not None

    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Create worker process
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            FOV_X,
            FOV_Y,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            WORKER_PERIOD,
            DETECTION_STRATEGY,
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
    detection1 = create_test_detection([200, 200, 400, 400], 1, 0.9)
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

    # Test 2: Test with multiple detections (should use HIGHEST_CONFIDENCE strategy)
    local_logger.info("--- Test 2: Processing multiple detections with HIGHEST_CONFIDENCE ---")
    detection_low_confidence = create_test_detection([100, 100, 200, 200], 1, 0.7)
    detection_high_confidence = create_test_detection([320, 320, 320, 320], 2, 0.95) # This one should be chosen

    simulate_detection_input(
        input_queue,
        [detection_low_confidence, detection_high_confidence],
        (0.0, 0.0, -100.0),  # 100 meters above ground
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have output for the detection with the highest confidence
    assert not output_queue.queue.empty()
    landing_info2 = output_queue.queue.get_nowait()
    assert landing_info2 is not None

    # To verify the correct detection was chosen, we can check the calculated angles.
    # The high confidence detection is at (320, 320), which is the center of the 640x640 image.
    # Therefore, the angles should be 0.
    assert landing_info2.angle_x == 0.0
    assert landing_info2.angle_y == 0.0
    local_logger.info("--- Test 2 Passed ---")

    # The case of "no detections" is handled by the worker's queue.get_nowait() exception.
    # No specific test is needed for an empty detection list as the data structure does not allow it.

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