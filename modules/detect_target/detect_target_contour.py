"""
Detects objects using the provided model.
"""

import time

import copy
import cv2
import numpy as np

from . import base_detect_target
from .. import image_and_time
from .. import detections_and_time

MIN_CONTOUR_AREA = 100
# some arbitrary value
UPPER_BLUE = np.array([130, 255, 255])
LOWER_BLUE = np.array([100, 50, 50])


class DetectTargetContour(base_detect_target.BaseDetectTarget):
    """
    Predicts annd locates landing pads using the Classical Computer Vision methodology.
    """

    def __init__(
        self,
        show_annotations: bool = False,
        save_name: str = "",
    ) -> None:
        """
        show_annotations: Display annotated images.
        save_name: filename prefix for logging detections and annotated images.
        """
        self.__counter = 0
        self.__show_annotations = show_annotations
        self.__filename_prefix = ""
        if save_name != "":
            self.__filename_prefix = save_name + "_" + str(int(time.time())) + "_"

    def detect_landing_pads_contours(
        self, image: np.ndarray, timestamp: float
    ) -> tuple[bool, detections_and_time.DetectionsAndTime | None, np.ndarray]:
        """
        Detects landing pads using contours/classical cv.
        image: Current image frame.
        timestamp: Timestamp for the detections.
        Return: Success, the DetectionsAndTime object, and the annotated image.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, LOWER_BLUE, UPPER_BLUE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return False, None, image

        # Create the DetectionsAndTime object
        result, detections = detections_and_time.DetectionsAndTime.create(timestamp)
        if not result:
            return False, None, image

        sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)
        image_annotated = copy.deepcopy(image)
        for i, contour in enumerate(sorted_contour):
            contour_area = cv2.contourArea(contour)

            if contour_area < MIN_CONTOUR_AREA:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            enclosing_area = np.pi * (radius**2)
            circularity = contour_area / enclosing_area

            if circularity < 0.7 or circularity > 1.3:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bounds = np.array([x, y, x + w, y + h])
            confidence, label = 1.0, 0

            # Create a Detection object and append it to detections
            result, detection = detections_and_time.Detection.create(bounds, label, confidence)
            if result:
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
    ) -> tuple[bool, detections_and_time.DetectionsAndTime] | tuple[False, None]:
        """
        Runs object detection on the provided image and returns the detections.
        data: Image with a timestamp.
        Return: Success and the detections.
        """
        image = data.image
        timestamp = data.timestamp

        result, detections, image_annotated = self.detect_landing_pads_contours(image, timestamp)

        if not result:
            return False, None

        # Logging
        if self.__filename_prefix != "":
            filename = self.__filename_prefix + str(self.__counter)
            # Object detections
            with open(filename + ".txt", "w", encoding="utf-8") as file:
                # Use internal string representation
                file.write(repr(detections))
            # Annotated image
            cv2.imwrite(filename + ".png", image_annotated)  # type: ignore
            self.__counter += 1
        if self.__show_annotations:
            cv2.imshow("Annotated", image_annotated)  # type: ignore
        return True, detections
