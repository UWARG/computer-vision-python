"""
Detects bright spots in images.
"""

import time

import cv2
import numpy as np

from . import base_detect_target
from .. import detections_and_time
from .. import image_and_time
from ..common.modules.logger import logger


# Label for brightspots; is 1 since 0 is used for blue landing pads
DETECTION_LABEL = 1
# SimpleBlobDetector is a binary detector, so a detection has confidence 1.0 by default
CONFIDENCE = 1.0


# Class has 15 attributes
# pylint: disable=too-many-instance-attributes
class DetectTargetBrightspotConfig:
    """
    Configuration for DetectTargetBrightspot.
    """

    def __init__(
        self,
        brightspot_percentile_threshold: float,
        filter_by_color: bool,
        blob_color: int,
        filter_by_circularity: bool,
        min_circularity: float,
        max_circularity: float,
        filter_by_inertia: bool,
        min_inertia_ratio: float,
        max_inertia_ratio: float,
        filter_by_convexity: bool,
        min_convexity: float,
        max_convexity: float,
        filter_by_area: bool,
        min_area_pixels: int,
        max_area_pixels: int,
    ) -> None:
        """
        Initializes the configuration for DetectTargetBrightspot.

        brightspot_percentile_threshold: Percentile threshold for bright spots.
        filter_by_color: Whether to filter by color.
        blob_color: Color of the blob.
        filter_by_circularity: Whether to filter by circularity.
        min_circularity: Minimum circularity.
        max_circularity: Maximum circularity.
        filter_by_inertia: Whether to filter by inertia.
        min_inertia_ratio: Minimum inertia ratio.
        max_inertia_ratio: Maximum inertia ratio.
        filter_by_convexity: Whether to filter by convexity.
        min_convexity: Minimum convexity.
        max_convexity: Maximum convexity.
        filter_by_area: Whether to filter by area.
        min_area_pixels: Minimum area in pixels.
        max_area_pixels: Maximum area in pixels.
        """
        self.brightspot_percentile_threshold = brightspot_percentile_threshold
        self.filter_by_color = filter_by_color
        self.blob_color = blob_color
        self.filter_by_circularity = filter_by_circularity
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.filter_by_inertia = filter_by_inertia
        self.min_inertia_ratio = min_inertia_ratio
        self.max_inertia_ratio = max_inertia_ratio
        self.filter_by_convexity = filter_by_convexity
        self.min_convexity = min_convexity
        self.max_convexity = max_convexity
        self.filter_by_area = filter_by_area
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels


# pylint: enable=too-many-instance-attributes


class DetectTargetBrightspot(base_detect_target.BaseDetectTarget):
    """
    Detects bright spots in images.
    """

    def __init__(
        self,
        config: DetectTargetBrightspotConfig,
        local_logger: logger.Logger,
        show_annotations: bool = False,
        save_name: str = "",
    ) -> None:
        """
        Initializes the bright spot detector.

        show_annotations: Display annotated images.
        save_name: Filename prefix for logging detections and annotated images.
        """
        self.__config = config
        self.__counter = 0
        self.__local_logger = local_logger
        self.__show_annotations = show_annotations
        self.__filename_prefix = ""
        if save_name != "":
            self.__filename_prefix = f"{save_name}_{int(time.time())}_"

    def run(
        self, data: image_and_time.ImageAndTime
    ) -> tuple[True, detections_and_time.DetectionsAndTime] | tuple[False, None]:
        """
        Runs brightspot detection on the provided image and returns the detections.

        data: Image with a timestamp.

        Return: Success, detections.
        """
        start_time = time.time()

        image = data.image
        try:
            grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Catching all exceptions for library call
        # pylint: disable-next=broad-exception-caught
        except Exception as exception:
            self.__local_logger.error(
                f"{time.time()}: Failed to convert to greyscale, exception: {exception}"
            )
            return False, None

        brightspot_threshold = np.percentile(
            grey_image, self.__config.brightspot_percentile_threshold
        )

        # Apply thresholding to isolate bright spots
        threshold_used, bw_image = cv2.threshold(
            grey_image, brightspot_threshold, 255, cv2.THRESH_BINARY
        )
        if threshold_used == 0:
            self.__local_logger.error(f"{time.time()}: Failed to threshold image.")
            return False, None

        # Set up SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = self.__config.filter_by_color
        params.blobColor = self.__config.blob_color
        params.filterByCircularity = self.__config.filter_by_circularity
        params.minCircularity = self.__config.min_circularity
        params.maxCircularity = self.__config.max_circularity
        params.filterByInertia = self.__config.filter_by_inertia
        params.minInertiaRatio = self.__config.min_inertia_ratio
        params.maxInertiaRatio = self.__config.max_inertia_ratio
        params.filterByConvexity = self.__config.filter_by_convexity
        params.minConvexity = self.__config.min_convexity
        params.maxConvexity = self.__config.max_convexity
        params.filterByArea = self.__config.filter_by_area
        params.minArea = self.__config.min_area_pixels
        params.maxArea = self.__config.max_area_pixels

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(bw_image)

        # A lack of detections is not an error, but should still not be forwarded
        if len(keypoints) == 0:
            self.__local_logger.info(f"{time.time()}: No brightspots detected.")
            return False, None

        # Annotate the image (green circle) with detected keypoints
        image_annotated = cv2.drawKeypoints(
            image, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Process bright spot detection
        result, detections = detections_and_time.DetectionsAndTime.create(data.timestamp)
        if not result:
            self.__local_logger.error(f"{time.time()}: Failed to create detections for image.")
            return False, None

        # Get Pylance to stop complaining
        assert detections is not None

        # Draw bounding boxes around detected keypoints
        for keypoint in keypoints:
            x, y = keypoint.pt
            size = keypoint.size
            bounds = np.array([x - size / 2, y - size / 2, x + size / 2, y + size / 2])
            result, detection = detections_and_time.Detection.create(
                bounds, DETECTION_LABEL, CONFIDENCE
            )
            if not result:
                self.__local_logger.error(f"{time.time()}: Failed to create bounding boxes.")
                return False, None

            # Get Pylance to stop complaining
            assert detections is not None

            detections.append(detection)

        # Logging is identical to detect_target_ultralytics.py
        # pylint: disable=duplicate-code
        end_time = time.time()

        # Logging
        self.__local_logger.info(
            f"{time.time()}: Count: {self.__counter}. Target detection took {end_time - start_time} seconds. Objects detected: {detections}."
        )

        if self.__filename_prefix != "":
            filename = self.__filename_prefix + str(self.__counter)

            # Annotated image
            cv2.imwrite(filename + ".png", image_annotated)  # type: ignore

            self.__counter += 1

        if self.__show_annotations:
            cv2.imshow("Annotated", image_annotated)  # type: ignore

        # pylint: enable=duplicate-code

        return True, detections
