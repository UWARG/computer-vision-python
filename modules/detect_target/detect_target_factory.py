"""
Factory pattern for constructing detect target class at runtime.
"""

import enum
import torch

from . import base_detect_target
from . import detect_target_ultralytics
from ..common.modules.logger import logger


class DetectTargetOption(enum.Enum):
    """
    ML for machine inference.
    """

    ML_ULTRALYTICS = 0


def create_detect_target(
    detect_target_option: DetectTargetOption,
    device: "str | int",
    model_path: str,
    override_full: bool,
    local_logger: logger.Logger,
    show_annotations: bool,
    save_name: str,
) -> tuple[bool, base_detect_target.BaseDetectTarget | None]:
    """
    Factory function to create a detection target object.

    Parameters:
    detect_target_option: Enumeration value to specify the type of detection.
    device: Target device for inference ("cpu" or CUDA device index).
    model_path: Path to the model file.
    override_full: Force full precision floating point calculations.
    local_logger: Logger instance for logging events.
    show_annotations: Whether to display annotated images.
    save_name: Prefix for saving logs or annotated images.

    Returns:
    Tuple containing success status and the instantiated detection object (if successful).
    """
    # Fall back to CPU if no GPU is available
    if device != "cpu" and not torch.cuda.is_available():
        local_logger.warning("CUDA not available. Falling back to CPU.")
        device = "cpu"

    match detect_target_option:
        case DetectTargetOption.ML_ULTRALYTICS:
            return True, detect_target_ultralytics.DetectTargetUltralytics(
                device,
                model_path,
                override_full,
                local_logger,
                show_annotations,
                save_name,
            )

    return False, None
