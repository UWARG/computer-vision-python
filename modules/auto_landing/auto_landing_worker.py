"""
Auto-landing worker.
"""

import os
import pathlib
import queue
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import auto_landing
from ..common.modules.logger import logger


def auto_landing_worker(
    fov_x: float,
    fov_y: float,
    im_h: float,
    im_w: float,
    period: float,
    detection_strategy: auto_landing.DetectionSelectionStrategy,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process for auto-landing operations.

    fov_x: Horizontal field of view in degrees.
    fov_y: Vertical field of view in degrees.
    im_h: Image height in pixels.
    im_w: Image width in pixels.
    period: Wait time in seconds between processing cycles.
    detection_strategy: Strategy for selecting detection when multiple targets are present.
    input_queue: Queue for receiving merged odometry detections.
    output_queue: Queue for sending auto-landing information.
    controller: Worker controller for pause/exit management.
    """

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    # Get Pylance to stop complaining
    assert local_logger is not None

    local_logger.info("Logger initialized", True)

    # Create auto-landing instance
    result, auto_lander = auto_landing.AutoLanding.create(
        fov_x, fov_y, im_h, im_w, local_logger, detection_strategy
    )
    if not result:
        local_logger.error("Worker failed to create AutoLanding object", True)
        return

    # Get Pylance to stop complaining
    assert auto_lander is not None

    local_logger.info("Auto-landing worker initialized successfully", True)

    while not controller.is_exit_requested():
        controller.check_pause()

        # Process detections if available
        input_data = None
        try:
            input_data = input_queue.queue.get_nowait()
        except queue.Empty:
            # No data available, continue
            continue

        if input_data is not None:
            result, landing_info = auto_lander.run(input_data)
            if result and landing_info:
                output_queue.queue.put(landing_info)

        time.sleep(period)
