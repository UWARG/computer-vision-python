"""
Factory pattern for constructing detect target class at runtime.
"""

import enum

from . import base_detect_target
from . import detect_target_brightspot
from . import detect_target_ultralytics
from ..common.modules.logger import logger


class DetectTargetOption(enum.Enum):
    """
    ML for machine inference.
    """

    ML_ULTRALYTICS = 0
    CV_BRIGHTSPOT = 1


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
    Construct detect target class at runtime.
    """
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
        case DetectTargetOption.CV_BRIGHTSPOT:
            return True, detect_target_brightspot.DetectTargetBrightspot(
                local_logger,
                show_annotations,
                save_name,
            )

    return False, None
