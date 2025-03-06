"""
Detects objects using the provided model.
"""

import time

import cv2
import numpy as np

from . import base_detect_target
from .. import image_and_time
from .. import detections_and_time
from ..common.modules.logger import logger


MIN_CONTOUR_AREA = 100
MAX_CIRCULARITY = 1.3
MIN_CIRCULARITY = 0.7
UPPER_BLUE = np.array([130, 255, 255])
LOWER_BLUE = np.array([100, 50, 50])
CONFIDENCE = 1.0
LABEL = 0


class DetectTargetContour(base_detect_target.BaseDetectTarget):
    """
    Predicts annd locates landing pads using the classical computer vision methodology.
    """

    def __init__(
        self, image_logger: logger.Logger, show_annotations: bool = False, save_name: str = ""
    ) -> None:
        """
        image_logger: Log annotated images.
        show_annotations: Display annotated images.
        save_name: filename prefix for logging detections and annotated images.
        """
        self.__counter = 0
        self.__show_annotations = show_annotations
        self.__filename_prefix = ""
        self.__logger = image_logger

        if save_name != "":
            self.__filename_prefix = save_name + "_" + str(int(time.time())) + "_"

    def detect_landing_pads_contours(
        self, image_and_time_data: image_and_time.ImageAndTime
    ) -> tuple[True, detections_and_time.DetectionsAndTime, np.ndarray] | tuple[False, None, None]:
        """
        Detects landing pads using contours/classical CV.

        image: Current image frame.
        timestamp: Timestamp for the detections.

        Return: Success, the DetectionsAndTime object, and the annotated image.
        """
        image = image_and_time_data.image
        timestamp = image_and_time_data.timestamp

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, LOWER_BLUE, UPPER_BLUE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return False, None, None

        result, detections = detections_and_time.DetectionsAndTime.create(timestamp)
        if not result:
            return False, None, None

        image_annotated = image
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)

            if contour_area < MIN_CONTOUR_AREA:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            enclosing_area = np.pi * (radius**2)
            circularity = contour_area / enclosing_area

            if circularity < MIN_CIRCULARITY or circularity > MAX_CIRCULARITY:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bounds = np.array([x, y, x + w, y + h])

            # Create a Detection object and append it to detections
            result, detection = detections_and_time.Detection.create(bounds, LABEL, CONFIDENCE)

            if not result:
                return False, None, None

            detections.append(detection)

            # Annotate the image
            cv2.rectangle(image_annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                image_annotated,
                f"landing-pad {i+1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        return True, detections, image_annotated

    def run(
        self, data: image_and_time.ImageAndTime
    ) -> tuple[True, detections_and_time.DetectionsAndTime] | tuple[False, None]:
        """
        Runs object detection on the provided image and returns the detections.

        data: Image with a timestamp.

        Return: Success and the detections.
        """

        result, detections, image_annotated = self.detect_landing_pads_contours(data)

        if not result:
            return False, None

        # Logging
        if self.__filename_prefix != "":
            filename = self.__filename_prefix + str(self.__counter)
            self.__logger.save_image(image_annotated, filename)
            self.__counter += 1

        if self.__show_annotations:
            cv2.imshow("Annotated", image_annotated)  # type: ignore

        return True, detections
