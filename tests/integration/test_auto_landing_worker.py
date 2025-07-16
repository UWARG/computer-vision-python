#!/usr/bin/env python3
"""
Integration test for auto_landing_worker.
Tests the worker process, queue operations, command processing, and lifecycle management.
"""

import multiprocessing as mp
import time
import queue
import pathlib
import sys
import os
import unittest.mock

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from modules.auto_landing import auto_landing_worker
from modules.auto_landing import auto_landing
from modules import merged_odometry_detections
from modules import detections_and_time
from modules.common.modules import orientation
from modules.common.modules import position_local
from modules.common.modules.mavlink import drone_odometry_local
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
import numpy as np


class MockLogger:
    """Mock logger that doesn't write to files."""

    def __init__(self, name: str) -> None:
        self.name = name

    def info(self, message: str, print_to_console: bool = False) -> None:
        if print_to_console:
            pass  # Suppress log output during testing

    def warning(self, message: str, print_to_console: bool = False) -> None:
        if print_to_console:
            pass  # Suppress log output during testing

    def error(self, message: str, print_to_console: bool = False) -> None:
        if print_to_console:
            pass  # Suppress log output during testing


def mock_logger_create(name: str, enable_file_logging: bool) -> tuple[bool, MockLogger]:
    """Mock logger create function."""
    return True, MockLogger(name)


class MockDetection:
    """Mock detection for testing."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9) -> None:
        self.x_1 = x1
        self.y_1 = y1
        self.x_2 = x2
        self.y_2 = y2
        self.confidence = confidence

    def get_centre(self) -> tuple[float, float]:
        """Return center coordinates."""
        center_x = (self.x_1 + self.x_2) / 2
        center_y = (self.y_1 + self.y_2) / 2
        return center_x, center_y


def create_mock_merged_detections(detections_list: list[tuple[float, float, float, float, float]], down_position: float = -10.0) -> merged_odometry_detections.MergedOdometryDetections:
    """Create mock merged detections for testing."""
    # Create drone position
    result, drone_position = position_local.PositionLocal.create(0.0, 0.0, down_position)
    if not result:
        raise RuntimeError("Failed to create drone position")
    assert drone_position is not None

    # Create drone orientation
    result, drone_orientation = orientation.Orientation.create(0.0, 0.0, 0.0)
    if not result:
        raise RuntimeError("Failed to create drone orientation")
    assert drone_orientation is not None

    # Create drone odometry
    result, drone_odometry = drone_odometry_local.DroneOdometryLocal.create(
        drone_position, drone_orientation
    )
    if not result:
        raise RuntimeError("Failed to create drone odometry")
    assert drone_odometry is not None

    # Create detections
    mock_detections = []
    for det_data in detections_list:
        mock_det = MockDetection(*det_data)
        mock_detections.append(mock_det)

    # Create merged detections
    result, merged = merged_odometry_detections.MergedOdometryDetections.create(
        drone_odometry, mock_detections
    )
    if not result:
        raise RuntimeError(
            f"Failed to create merged detections with {len(mock_detections)} detections"
        )
    assert merged is not None

    return merged


def test_basic_worker_functionality() -> bool:
    """Test basic worker functionality - processing detections."""
    print("=== Testing Basic Worker Functionality ===")

    # Setup
    controller = worker_controller.WorkerController()
    mp_manager = mp.Manager()

    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Worker parameters
    fov_x = 60.0
    fov_y = 45.0
    im_h = 480
    im_w = 640
    period = 0.1
    detection_strategy = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER

    # Start worker
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            fov_x,
            fov_y,
            im_h,
            im_w,
            period,
            detection_strategy,
            input_queue,
            output_queue,
            command_queue,
            controller,
        ),
    )

    print("Starting worker...")
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Enable auto-landing
    enable_command = auto_landing_worker.AutoLandingCommand("enable")
    command_queue.queue.put(enable_command)

    # Wait for command to be processed
    time.sleep(0.2)

    # Send test detection
    detections = [(100, 100, 200, 200, 0.9)]  # x1, y1, x2, y2, confidence
    mock_merged = create_mock_merged_detections(detections)
    input_queue.queue.put(mock_merged)

    # Wait for processing
    time.sleep(0.5)

    # Check output
    try:
        landing_info = output_queue.queue.get_nowait()
        assert landing_info is not None
        assert hasattr(landing_info, "angle_x")
        assert hasattr(landing_info, "angle_y")
        assert hasattr(landing_info, "target_dist")
        print(f"✅ Worker processed detection successfully")
        print(f"   Angle X: {landing_info.angle_x:.4f}")
        print(f"   Angle Y: {landing_info.angle_y:.4f}")
        print(f"   Target Distance: {landing_info.target_dist:.4f}")
    except queue.Empty:
        print("❌ No output received from worker")
        assert False, "Worker should have produced output"

    # Cleanup
    controller.request_exit()
    input_queue.fill_and_drain_queue()
    output_queue.fill_and_drain_queue()
    command_queue.fill_and_drain_queue()
    worker.join()

    print("✅ Basic worker functionality test passed")
    return True


def test_worker_commands() -> bool:
    """Test worker command processing (enable/disable)."""
    print("\n=== Testing Worker Commands ===")

    # Setup
    controller = worker_controller.WorkerController()
    mp_manager = mp.Manager()

    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Worker parameters
    fov_x = 60.0
    fov_y = 45.0
    im_h = 480
    im_w = 640
    period = 0.1
    detection_strategy = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER

    # Start worker
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            fov_x,
            fov_y,
            im_h,
            im_w,
            period,
            detection_strategy,
            input_queue,
            output_queue,
            command_queue,
            controller,
        ),
    )

    print("Starting worker...")
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Test disabled state (default)
    print("Testing disabled state...")
    detections = [(100, 100, 200, 200, 0.9)]
    mock_merged = create_mock_merged_detections(detections)
    input_queue.queue.put(mock_merged)

    time.sleep(0.3)

    # Should not produce output when disabled
    try:
        output_queue.queue.get_nowait()
        print("❌ Worker should not process detections when disabled")
        assert False, "Worker should not process detections when disabled"
    except queue.Empty:
        print("✅ Worker correctly ignores detections when disabled")

    # Test enable command
    print("Testing enable command...")
    enable_command = auto_landing_worker.AutoLandingCommand("enable")
    command_queue.queue.put(enable_command)

    time.sleep(0.2)

    # Send detection after enabling
    input_queue.queue.put(mock_merged)

    time.sleep(0.3)

    # Should produce output when enabled
    try:
        landing_info = output_queue.queue.get_nowait()
        assert landing_info is not None
        print("✅ Worker processes detections when enabled")
    except queue.Empty:
        print("❌ Worker should process detections when enabled")
        assert False, "Worker should process detections when enabled"

    # Test disable command
    print("Testing disable command...")
    disable_command = auto_landing_worker.AutoLandingCommand("disable")
    command_queue.queue.put(disable_command)

    time.sleep(0.2)

    # Send detection after disabling
    input_queue.queue.put(mock_merged)

    time.sleep(0.3)

    # Should not produce output when disabled
    try:
        output_queue.queue.get_nowait()
        print("❌ Worker should not process detections after disable")
        assert False, "Worker should not process detections after disable"
    except queue.Empty:
        print("✅ Worker correctly stops processing after disable")

    # Cleanup
    controller.request_exit()
    input_queue.fill_and_drain_queue()
    output_queue.fill_and_drain_queue()
    command_queue.fill_and_drain_queue()
    worker.join()

    print("✅ Worker commands test passed")
    return True


def test_worker_no_detections() -> bool:
    """Test worker behavior with no detections."""
    print("\n=== Testing Worker with No Detections ===")

    try:
        # Setup
        controller = worker_controller.WorkerController()
        mp_manager = mp.Manager()

        input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
        output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
        command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

        # Worker parameters
        fov_x = 60.0
        fov_y = 45.0
        im_h = 480
        im_w = 640
        period = 0.1
        detection_strategy = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER

        # Start worker
        worker = mp.Process(
            target=auto_landing_worker.auto_landing_worker,
            args=(
                fov_x,
                fov_y,
                im_h,
                im_w,
                period,
                detection_strategy,
                input_queue,
                output_queue,
                command_queue,
                controller,
            ),
        )

        print("Starting worker...")
        worker.start()

        # Give worker time to initialize
        time.sleep(0.5)

        # Verify worker is alive
        if not worker.is_alive():
            print("❌ Worker failed to start")
            return False

        # Enable auto-landing
        enable_command = auto_landing_worker.AutoLandingCommand("enable")
        command_queue.queue.put(enable_command)

        time.sleep(0.2)

        # Don't send any detections (this simulates the real scenario where
        # the data merge worker doesn't produce MergedOdometryDetections when there are no detections)
        print("Not sending any detections to worker (simulating no detections scenario)...")

        print("Waiting to ensure worker doesn't produce output...")
        time.sleep(0.3)

        # Should not produce output when no data is sent
        print("Checking if worker produced output...")
        try:
            result = output_queue.queue.get_nowait()
            print(f"❌ Worker should not produce output when no data is sent, but got: {result}")
            return False
        except queue.Empty:
            print("✅ Worker correctly handles no input data")

        # Cleanup
        controller.request_exit()
        input_queue.fill_and_drain_queue()
        output_queue.fill_and_drain_queue()
        command_queue.fill_and_drain_queue()
        worker.join()

        print("✅ No detections test passed")
        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        # Try to cleanup if possible
        try:
            controller.request_exit()
            worker.join(timeout=1.0)
        except:
            pass
        return False


def test_worker_multiple_detections() -> bool:
    """Test worker with multiple detections and different strategies."""
    print("\n=== Testing Worker with Multiple Detections ===")

    strategies = [
        auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER,
        auto_landing.DetectionSelectionStrategy.LARGEST_AREA,
        auto_landing.DetectionSelectionStrategy.HIGHEST_CONFIDENCE,
    ]

    for strategy in strategies:
        print(f"Testing strategy: {strategy.value}")

        # Setup
        controller = worker_controller.WorkerController()
        mp_manager = mp.Manager()

        input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
        output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
        command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

        # Worker parameters
        fov_x = 60.0
        fov_y = 45.0
        im_h = 480
        im_w = 640
        period = 0.1

        # Start worker
        worker = mp.Process(
            target=auto_landing_worker.auto_landing_worker,
            args=(
                fov_x,
                fov_y,
                im_h,
                im_w,
                period,
                strategy,
                input_queue,
                output_queue,
                command_queue,
                controller,
            ),
        )

        worker.start()
        time.sleep(0.5)

        # Enable auto-landing
        enable_command = auto_landing_worker.AutoLandingCommand("enable")
        command_queue.queue.put(enable_command)
        time.sleep(0.2)

        # Send multiple detections
        detections = [
            (50, 50, 100, 100, 0.7),  # Top-left, lower confidence
            (300, 200, 400, 300, 0.9),  # Center-right, high confidence
            (200, 180, 300, 280, 0.8),  # Near center, medium confidence
        ]
        mock_merged = create_mock_merged_detections(detections)
        input_queue.queue.put(mock_merged)

        time.sleep(0.5)

        # Should produce output
        try:
            landing_info = output_queue.queue.get_nowait()
            assert landing_info is not None
            print(f"✅ Strategy {strategy.value} produced output")
        except queue.Empty:
            print(f"❌ Strategy {strategy.value} failed to produce output")
            assert False, f"Strategy {strategy.value} should produce output"

        # Cleanup
        controller.request_exit()
        input_queue.fill_and_drain_queue()
        output_queue.fill_and_drain_queue()
        command_queue.fill_and_drain_queue()
        worker.join()

    print("✅ Multiple detections test passed")
    return True


def test_worker_lifecycle() -> bool:
    """Test worker lifecycle management (pause, resume, exit)."""
    print("\n=== Testing Worker Lifecycle ===")

    # Setup
    controller = worker_controller.WorkerController()
    mp_manager = mp.Manager()

    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Worker parameters
    fov_x = 60.0
    fov_y = 45.0
    im_h = 480
    im_w = 640
    period = 0.1
    detection_strategy = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER

    # Start worker
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            fov_x,
            fov_y,
            im_h,
            im_w,
            period,
            detection_strategy,
            input_queue,
            output_queue,
            command_queue,
            controller,
        ),
    )

    print("Starting worker...")
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Verify worker is alive
    assert worker.is_alive(), "Worker should be alive after start"
    print("✅ Worker started successfully")

    # Test pause
    print("Testing pause...")
    controller.request_pause()
    time.sleep(0.2)

    # Worker should still be alive but paused
    assert worker.is_alive(), "Worker should still be alive when paused"
    print("✅ Worker paused successfully")

    # Test resume
    print("Testing resume...")
    controller.request_resume()
    time.sleep(0.2)

    # Worker should still be alive and resumed
    assert worker.is_alive(), "Worker should still be alive after resume"
    print("✅ Worker resumed successfully")

    # Test exit
    print("Testing exit...")
    controller.request_exit()

    # Give worker time to exit gracefully
    worker.join(timeout=2.0)

    # Worker should have exited
    assert not worker.is_alive(), "Worker should have exited"
    print("✅ Worker exited successfully")

    # Cleanup
    input_queue.fill_and_drain_queue()
    output_queue.fill_and_drain_queue()
    command_queue.fill_and_drain_queue()

    print("✅ Worker lifecycle test passed")
    return True


def test_worker_invalid_commands() -> bool:
    """Test worker behavior with invalid commands."""
    print("\n=== Testing Worker with Invalid Commands ===")

    # Setup
    controller = worker_controller.WorkerController()
    mp_manager = mp.Manager()

    input_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    output_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    command_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    # Worker parameters
    fov_x = 60.0
    fov_y = 45.0
    im_h = 480
    im_w = 640
    period = 0.1
    detection_strategy = auto_landing.DetectionSelectionStrategy.NEAREST_TO_CENTER

    # Start worker
    worker = mp.Process(
        target=auto_landing_worker.auto_landing_worker,
        args=(
            fov_x,
            fov_y,
            im_h,
            im_w,
            period,
            detection_strategy,
            input_queue,
            output_queue,
            command_queue,
            controller,
        ),
    )

    print("Starting worker...")
    worker.start()

    # Give worker time to initialize
    time.sleep(0.5)

    # Test invalid command
    print("Testing invalid command...")
    invalid_command = auto_landing_worker.AutoLandingCommand("invalid_command")
    command_queue.queue.put(invalid_command)

    time.sleep(0.3)

    # Worker should still be alive after invalid command
    assert worker.is_alive(), "Worker should still be alive after invalid command"
    print("✅ Worker handled invalid command gracefully")

    # Test invalid command type
    print("Testing invalid command type...")
    command_queue.queue.put("not_a_command_object")

    time.sleep(0.3)

    # Worker should still be alive after invalid command type
    assert worker.is_alive(), "Worker should still be alive after invalid command type"
    print("✅ Worker handled invalid command type gracefully")

    # Cleanup
    controller.request_exit()
    input_queue.fill_and_drain_queue()
    output_queue.fill_and_drain_queue()
    command_queue.fill_and_drain_queue()
    worker.join()

    print("✅ Invalid commands test passed")
    return True


def main() -> int:
    """Run all tests."""
    print("Starting Auto Landing Worker Integration Tests...")

    # Create logs directory structure required by logger
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create timestamp-based subdirectory (logger expects this)
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = logs_dir / timestamp
    session_dir.mkdir(exist_ok=True)

    # Clean up any existing log files from previous test runs
    for log_file in session_dir.glob("*auto_landing_worker*"):
        try:
            log_file.unlink()
        except:
            pass  # Ignore errors if file doesn't exist or is locked

    tests = [
        test_basic_worker_functionality,
        test_worker_commands,
        test_worker_no_detections,
        test_worker_multiple_detections,
        test_worker_lifecycle,
        test_worker_invalid_commands,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    # Clean up log files after tests
    for log_file in session_dir.glob("*auto_landing_worker*"):
        try:
            log_file.unlink()
        except:
            pass  # Ignore errors if file doesn't exist or is locked

    # Clean up session directory if empty
    try:
        session_dir.rmdir()
    except:
        pass  # Ignore if directory is not empty or doesn't exist

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
