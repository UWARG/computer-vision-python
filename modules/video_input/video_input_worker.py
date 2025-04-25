"""
Gets images from the camera.
"""

import os
import pathlib
import time
import cv2

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import video_input
from ..common.modules.camera import camera_factory
from ..common.modules.camera import camera_opencv
from ..common.modules.camera import camera_picamera2
from ..common.modules.logger import logger


def video_input_worker(
    camera_option: camera_factory.CameraOption,
    width: int,
    height: int,
    camera_config: camera_opencv.ConfigOpenCV | camera_picamera2.ConfigPiCamera2,
    maybe_image_name: str | None,
    period: float,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    period is the minimum period between image captures in seconds.
    output_queue is the data queue.
    controller is how the main process communicates to this worker process.
    """
    setup_start_time = time.time()

    worker_name = pathlib.Path(__file__).stem
    process_id = os.getpid()
    result, local_logger = logger.Logger.create(f"{worker_name}_{process_id}", True)
    if not result:
        print("ERROR: Worker failed to create logger")
        return

    assert local_logger is not None

    local_logger.info("Logger initialized")

    result, input_device = video_input.VideoInput.create(
        camera_option, width, height, camera_config, maybe_image_name, local_logger
    )
    if not result:
        local_logger.error("Worker failed to create class object")
        return

    # Get Pylance to stop complaining
    assert input_device is not None

    setup_end_time = time.time()

    local_logger.info(
        f"{time.time()}: Worker setup took {setup_end_time - setup_start_time} seconds."
    )

    while not controller.is_exit_requested():
        iteration_start_time = time.time()

        controller.check_pause()

        while time.time() - iteration_start_time < period:  # BE CAREFUL HERE! UNITS MAY MISMATCH
            result, image = input_device.get_raw_image()
            cv2.imshow("Camera feed", image)
            
            if cv2.waitKey(20) & 0xFF == ord("q"): break
  
        result, value = input_device.run()
        if not result:
            continue

        output_queue.queue.put(value)

        iteration_end_time = time.time()

        local_logger.info(
            f"{time.time()}: Worker iteration took {iteration_end_time - iteration_start_time} seconds."
        )
        
    cv2.destroyAllWindows()