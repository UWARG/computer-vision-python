"""
Gets images from the camera.
"""

import os
import pathlib
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import video_input
from ..common.modules.camera import camera_factory
from ..common.modules.camera import camera_opencv
from ..common.modules.camera import camera_picamera2
from ..common.modules.logger import logger


def video_input_worker(
    period: int,
    camera_option: camera_factory.CameraOption,
    width: int,
    height: int,
    camera_config: camera_opencv.ConfigOpenCV | camera_picamera2.ConfigPiCamera2,
    save_prefix: str,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    period is the minimum period between image captures.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    assert local_logger is not None

    local_logger.info("Logger initialized")

    result, input_device = video_input.VideoInput.create(
        camera_option, width, height, camera_config, save_prefix, local_logger
    )
    if not result:
        local_logger.error("Worker failed to create class object")
        return

    # Get Pylance to stop complaining
    assert input_device is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        time.sleep(period)

        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)
