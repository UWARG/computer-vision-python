"""
Auto-landing worker.
"""

import os
import pathlib
import time

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from . import auto_landing
from ..common.modules.logger import logger


class AutoLandingCommand:
    """
    Command structure for controlling auto-landing operations.
    """
    def __init__(self, command: str):
        self.command = command


def auto_landing_worker(
    fov_x: float,
    fov_y: float,
    im_h: float,
    im_w: float,
    period: float,
    detection_strategy: auto_landing.DetectionSelectionStrategy,
    input_queue: queue_proxy_wrapper.QueueProxyWrapper,
    output_queue: queue_proxy_wrapper.QueueProxyWrapper,
    command_queue: queue_proxy_wrapper.QueueProxyWrapper,
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
    command_queue: Queue for receiving enable/disable commands.
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

    # Create auto-landing controller
    result, landing_controller = auto_landing.AutoLandingController.create(
        auto_lander, local_logger
    )
    if not result:
        local_logger.error("Worker failed to create AutoLandingController object", True)
        return

    # Get Pylance to stop complaining
    assert landing_controller is not None

    local_logger.info("Auto-landing worker initialized successfully", True)

    while not controller.is_exit_requested():
        controller.check_pause()

        # Process commands first
        _process_commands(command_queue, landing_controller, local_logger)

        # Process detections if available
        input_data = None
        try:
            input_data = input_queue.queue.get_nowait()
        except:
            # No data available, continue
            pass

        if input_data is not None:
            result, landing_info = landing_controller.process_detections(input_data)
            if result and landing_info:
                output_queue.queue.put(landing_info)

        time.sleep(period)


def _process_commands(
    command_queue: queue_proxy_wrapper.QueueProxyWrapper,
    landing_controller: auto_landing.AutoLandingController,
    local_logger: logger.Logger,
) -> None:
    """
    Process all available commands in the command queue.
    
    command_queue: Queue containing AutoLandingCommand objects.
    landing_controller: Controller instance to execute commands on.
    local_logger: Logger for command processing.
    """
    while True:
        try:
            command = command_queue.queue.get_nowait()
            if command is None:
                break
                
            if isinstance(command, AutoLandingCommand):
                _execute_command(command, landing_controller, local_logger)
            else:
                local_logger.warning(f"Received invalid command type: {type(command)}", True)
                
        except:
            # No more commands available
            break


def _execute_command(
    command: AutoLandingCommand,
    landing_controller: auto_landing.AutoLandingController,
    local_logger: logger.Logger,
) -> None:
    """
    Execute an auto-landing command.
    
    command: Command to execute.
    landing_controller: Controller instance to execute command on.
    local_logger: Logger for command execution.
    """
    local_logger.info(f"Executing command: {command.command}", True)
    
    success = False
    if command.command == "enable":
        success = landing_controller.enable()
    elif command.command == "disable":
        success = landing_controller.disable()
    else:
        local_logger.error(f"Unknown command: {command.command}. Only 'enable' and 'disable' are supported.", True)
        return
    
    if success:
        local_logger.info(f"Command '{command.command}' executed successfully", True)
    else:
        local_logger.error(f"Command '{command.command}' failed to execute", True)
