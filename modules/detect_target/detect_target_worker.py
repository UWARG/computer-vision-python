"""
Gets frames and outputs detections in image space.
"""

import os
import pathlib

from modules import image_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import detect_target_brightspot
from . import detect_target_factory
from . import detect_target_ultralytics
from ..common.modules.logger import logger


def detect_target_worker(
    save_name: str,
    show_annotations: bool,
    detect_target_option: detect_target_factory.DetectTargetOption,
    config: (
        detect_target_brightspot.DetectTargetBrightspotConfig
        | detect_target_ultralytics.DetectTargetUltralyticsConfig
    ),
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    controller: worker_controller.WorkerController,
) -> None:
    """
    Worker process.

    detect_target_option: Enumeration to construct the appropriate object.
    device, model_path, override_full, show_annotations, and save_name are initial settings.
    input_queue and output_queue are data queues.
    controller is how the main process communicates to this worker process.
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

    result, detector = detect_target_factory.create_detect_target(
        save_name,
        show_annotations,
        detect_target_option,
        config,
        local_logger,
    )

    if not result:
        local_logger.error("Could not construct detector.")
        return

    # Get Pylance to stop complaining
    assert detector is not None

    while not controller.is_exit_requested():
        controller.check_pause()

        input_data = input_queue.queue.get()
        if input_data is None:
            local_logger.info("Recieved type None, exiting.")
            break

        if not isinstance(input_data, image_and_time.ImageAndTime):
            local_logger.warning(f"Skipping unexpected input: {input_data}")
            continue

        result, value = detector.run(input_data)
        if not result:
            continue

        output_queue.queue.put(value)
