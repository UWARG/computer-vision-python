""" 
Auto-landing script that calculates the necessary parameters 
for use with LANDING_TARGET MAVLink command. 
"""

import math
from enum import Enum
from typing import Optional
import threading

from .. import merged_odometry_detections
from ..common.modules.logger import logger


class DetectionSelectionStrategy(Enum):
    """
    Strategies for selecting which detection to use when multiple targets are detected.
    """

    NEAREST_TO_CENTER = "nearest_to_center"  # Choose detection closest to image center
    LARGEST_AREA = "largest_area"  # Choose detection with largest bounding box area
    HIGHEST_CONFIDENCE = "highest_confidence"  # Choose detection with highest confidence
    FIRST_DETECTION = "first_detection"  # Use first detection in list (original behavior)


class AutoLandingInformation:
    """
    Information necessary for the LANDING_TARGET MAVLink command.
    """

    def __init__(self, angle_x: float, angle_y: float, target_dist: float) -> None:
        """
        Information necessary for the LANDING_TARGET MAVLink command.

        angle_x: Angle of the x coordinate of the bounding box within -π to π (rads).
        angle_y: Angle of the y coordinate of the bounding box within -π to π (rads).
        target_dist: Horizontal distance of vehicle to target (meters).
        """

        self.angle_x = angle_x
        self.angle_y = angle_y
        self.target_dist = target_dist


class AutoLanding:
    """
    Auto-landing script that calculates the necessary parameters
    for use with LANDING_TARGET MAVLink command.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        local_logger: logger.Logger,
        selection_strategy: DetectionSelectionStrategy = DetectionSelectionStrategy.NEAREST_TO_CENTER,
    ) -> "tuple [bool, AutoLanding | None ]":
        """
        fov_x: The horizontal camera field of view in degrees.
        fov_y: The vertical camera field of view in degrees.
        im_w: Width of image.
        im_h: Height of image.
        selection_strategy: Strategy for selecting which detection to use when multiple targets are detected.

        Returns an AutoLanding object.
        """

        return True, AutoLanding(
            cls.__create_key, fov_x, fov_y, im_h, im_w, local_logger, selection_strategy
        )

    def __init__(
        self,
        class_private_create_key: object,
        fov_x: float,
        fov_y: float,
        im_h: float,
        im_w: float,
        local_logger: logger.Logger,
        selection_strategy: DetectionSelectionStrategy,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is AutoLanding.__create_key, "Use create() method"

        self.fov_x = fov_x
        self.fov_y = fov_y
        self.im_h = im_h
        self.im_w = im_w
        self.__logger = local_logger
        self.__selection_strategy = selection_strategy

    def _select_detection(self, detections: list) -> Optional[int]:
        """
        Select which detection to use based on the configured strategy.

        Returns the index of the selected detection, or None if no suitable detection found.
        """
        if not detections:
            return None

        if len(detections) == 1:
            return 0

        if self.__selection_strategy == DetectionSelectionStrategy.FIRST_DETECTION:
            return 0

        if self.__selection_strategy == DetectionSelectionStrategy.NEAREST_TO_CENTER:
            # Find detection closest to image center
            image_center_x = self.im_w / 2
            image_center_y = self.im_h / 2

            min_distance = float("inf")
            selected_index = 0

            for i, detection in enumerate(detections):
                det_center_x, det_center_y = detection.get_centre()
                distance = math.sqrt(
                    (det_center_x - image_center_x) ** 2 + (det_center_y - image_center_y) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    selected_index = i

            return selected_index

        elif self.__selection_strategy == DetectionSelectionStrategy.LARGEST_AREA:
            # Find detection with largest bounding box area
            max_area = 0
            selected_index = 0

            for i, detection in enumerate(detections):
                width = detection.x_2 - detection.x_1
                height = detection.y_2 - detection.y_1
                area = width * height
                if area > max_area:
                    max_area = area
                    selected_index = i

            return selected_index

        elif self.__selection_strategy == DetectionSelectionStrategy.HIGHEST_CONFIDENCE:
            # Find detection with highest confidence
            max_confidence = 0
            selected_index = 0

            for i, detection in enumerate(detections):
                if detection.confidence > max_confidence:
                    max_confidence = detection.confidence
                    selected_index = i

            return selected_index

        # Default fallback
        return 0

    def run(
        self, odometry_detections: merged_odometry_detections.MergedOdometryDetections
    ) -> "tuple[bool, AutoLandingInformation | None]":
        """
        Calculates the x and y angles in radians of the bounding box based on its center.

        odometry_detections: A merged odometry dectections object.

        Returns an AutoLandingInformation object.
        """

        # Check if we have any detections
        if not odometry_detections.detections:
            self.__logger.warning("No detections available for auto-landing", True)
            return False, None

        # Select which detection to use
        selected_index = self._select_detection(odometry_detections.detections)
        if selected_index is None:
            self.__logger.error("Failed to select detection for auto-landing", True)
            return False, None

        selected_detection = odometry_detections.detections[selected_index]

        # Log selection information
        self.__logger.info(
            f"Selected detection {selected_index + 1}/{len(odometry_detections.detections)} using strategy: {self.__selection_strategy.value}",
            True,
        )

        x_center, y_center = selected_detection.get_centre()

        angle_x = (x_center - self.im_w / 2) * (self.fov_x * (math.pi / 180)) / self.im_w
        angle_y = (y_center - self.im_h / 2) * (self.fov_y * (math.pi / 180)) / self.im_h

        # This is height above ground level (AGL)
        # down is how many meters down you are from home position, which is on the ground
        height_agl = odometry_detections.odometry_local.position.down * -1

        x_dist = math.tan(angle_x) * height_agl
        y_dist = math.tan(angle_y) * height_agl
        ground_hyp = (x_dist**2 + y_dist**2) ** 0.5
        target_to_vehicle_dist = (ground_hyp**2 + height_agl**2) ** 0.5

        self.__logger.info(
            f"X angle: {angle_x} Y angle: {angle_y}\nRequired horizontal correction: {ground_hyp} Distance from vehicle to target: {target_to_vehicle_dist}",
            True,
        )

        return True, AutoLandingInformation(angle_x, angle_y, target_to_vehicle_dist)


class AutoLandingController:
    """
    Simple controller for turning auto-landing on/off.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        auto_landing: AutoLanding,
        local_logger: logger.Logger,
    ) -> "tuple[bool, AutoLandingController | None]":
        """
        Create an AutoLandingController instance.

        auto_landing: The AutoLanding instance to control.
        local_logger: Logger instance for logging.

        Returns an AutoLandingController object.
        """
        return True, AutoLandingController(cls.__create_key, auto_landing, local_logger)

    def __init__(
        self,
        class_private_create_key: object,
        auto_landing: AutoLanding,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is AutoLandingController.__create_key, "Use create() method"

        self.__auto_landing = auto_landing
        self.__logger = local_logger
        self.__enabled = False
        self.__enabled_lock = threading.Lock()

    def is_enabled(self) -> bool:
        """
        Check if auto-landing is enabled.
        """
        with self.__enabled_lock:
            return self.__enabled

    def enable(self) -> bool:
        """
        Enable auto-landing system.

        Returns True if successfully enabled, False otherwise.
        """
        with self.__enabled_lock:
            if not self.__enabled:
                self.__enabled = True
                self.__logger.info("Auto-landing system enabled", True)
                return True
            self.__logger.warning("Auto-landing system already enabled", True)
            return False

    def disable(self) -> bool:
        """
        Disable auto-landing system.

        Returns True if successfully disabled, False otherwise.
        """
        with self.__enabled_lock:
            if self.__enabled:
                self.__enabled = False
                self.__logger.info("Auto-landing system disabled", True)
                return True
            self.__logger.warning("Auto-landing system already disabled", True)
            return False

    def process_detections(
        self, odometry_detections: merged_odometry_detections.MergedOdometryDetections
    ) -> "tuple[bool, AutoLandingInformation | None]":
        """
        Process detections if auto-landing is enabled.

        Returns landing information if processing was successful, None otherwise.
        """
        with self.__enabled_lock:
            if not self.__enabled:
                return False, None

        # Process the detections using the auto-landing module
        result, landing_info = self.__auto_landing.run(odometry_detections)
        return result, landing_info
