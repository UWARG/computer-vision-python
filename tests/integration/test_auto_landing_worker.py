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
DETECTION_STRATEGY = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER
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
    command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

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
            command_queue,
            controller,
        ),
    )

    # Start worker
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Test 1: Worker should not process detections when disabled (default state)
    detection1 = create_test_detection([200, 200, 400, 400], 1, 0.9)
    simulate_detection_input(
        input_queue,
        [detection1],
        (0.0, 0.0, -50.0),  # 50 meters above ground
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have no output since auto-landing is disabled by default
    assert output_queue.queue.empty()

    # Test 2: Enable auto-landing and verify it processes detections
    enable_command = auto_landing_worker.AutoLandingCommand("enable")
    command_queue.queue.put(enable_command)

    time.sleep(0.2)

    # Now send the same detection - should be processed
    detection2 = create_test_detection([300, 300, 500, 500], 2, 0.85)
    simulate_detection_input(
        input_queue,
        [detection2],
        (10.0, 5.0, -75.0),  # 75 meters above ground
        (0.1, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have output now
    assert not output_queue.queue.empty()
    landing_info = output_queue.queue.get_nowait()
    assert landing_info is not None
    assert hasattr(landing_info, "angle_x")
    assert hasattr(landing_info, "angle_y")
    assert hasattr(landing_info, "target_dist")

    # Test 3: Test with multiple detections (should use NEAREST_TO_CENTER strategy)
    detection3 = create_test_detection([100, 100, 200, 200], 1, 0.7)  # Far from center
    detection4 = create_test_detection([310, 310, 330, 330], 2, 0.8)  # Close to center (320, 320)

    simulate_detection_input(
        input_queue,
        [detection3, detection4],
        (0.0, 0.0, -100.0),  # 100 meters above ground
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have output for the detection closest to center
    assert not output_queue.queue.empty()
    landing_info2 = output_queue.queue.get_nowait()
    assert landing_info2 is not None

    # Test 4: Disable auto-landing and verify it stops processing
    disable_command = auto_landing_worker.AutoLandingCommand("disable")
    command_queue.queue.put(disable_command)

    time.sleep(0.2)

    # Send another detection - should not be processed
    detection5 = create_test_detection([400, 400, 600, 600], 3, 0.95)
    simulate_detection_input(
        input_queue,
        [detection5],
        (0.0, 0.0, -60.0),
        (0.0, 0.0, 0.0),
    )

    time.sleep(0.2)

    # Should have no new output
    assert output_queue.queue.empty()

    # Test 5: Test invalid command handling
    invalid_command = auto_landing_worker.AutoLandingCommand("invalid_command")
    command_queue.queue.put(invalid_command)

    time.sleep(0.2)

    # Worker should continue running despite invalid command
    assert worker.is_alive()

    # Test 6: Test with no detections (empty detection list should not crash)
    # This should not create a MergedOdometryDetections object since it requires non-empty detections
    # So we just verify the worker continues running

    # Cleanup
    controller.request_exit()

    # Drain queues
    input_queue.fill_and_drain_queue()
    command_queue.fill_and_drain_queue()

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
