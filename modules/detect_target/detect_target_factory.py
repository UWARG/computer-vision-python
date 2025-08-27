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
    save_name: str,
    show_annotations: bool,
    detect_target_option: DetectTargetOption,
    config: (
        detect_target_brightspot.DetectTargetBrightspotConfig
        | detect_target_ultralytics.DetectTargetUltralyticsConfig
    ),
    local_logger: logger.Logger,
) -> tuple[bool, base_detect_target.BaseDetectTarget | None]:
    """
    Factory function to create a detection target object.

    Return:
    Success, detect target object.
    """
    match detect_target_option:
        case DetectTargetOption.ML_ULTRALYTICS:
            return True, detect_target_ultralytics.DetectTargetUltralytics(
                config,
                local_logger,
                show_annotations,
                save_name,
            )
        case DetectTargetOption.CV_BRIGHTSPOT:
            return True, detect_target_brightspot.DetectTargetBrightspot(
                config,
                local_logger,
                show_annotations,
                save_name,
            )

    return False, None
