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


class DetectTargetContour(base_detect_target.BaseDetectTarget):
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

    @staticmethod
    def is_contour_circular(contour: np.ndarray) -> bool:
        """
        Helper function for detect_landing_pads_contours.
        Checks if the shape is close to circular.
        Return: True is the shape is circular, false if it is not.
        """
        contour_minimum = 0.8
        perimeter = cv2.arcLength(contour, True)
        # Check if the perimeter is zero
        if perimeter == 0.0:
            return False

        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        return circularity > contour_minimum

    @staticmethod
    def is_contour_large_enough(contour: np.ndarray, min_diameter: float) -> bool:
        """
        Helper function for detect_landing_pads_contours.
        Checks if the shape is larger than the provided diameter.
        Return: True if it is, false if it not.
        """
        _, radius = cv2.minEnclosingCircle(contour)
        diameter = radius * 2
        return diameter >= min_diameter

    def detect_landing_pads_contours(
        self, image: "np.ndarray", timestamp: float
    ) -> "tuple[bool, detections_and_time.DetectionsAndTime | None, np.ndarray]":
        """
        Detects landing pads using contours/classical cv.
        image: Current image frame.
        timestamp: Timestamp for the detections.
        Return: Success, the DetectionsAndTime object, and the annotated image.
        """
        kernel = np.ones((2, 2), np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = 180
        im_bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
        im_dilation = cv2.dilate(im_bw, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(im_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return False, None, image

        contours_with_children = set(i for i, hier in enumerate(hierarchy[0]) if hier[2] != -1)
        parent_circular_contours = [
            cnt
            for i, cnt in enumerate(contours)
            if self.is_contour_circular(cnt)
            and self.is_contour_large_enough(cnt, 7)
            and i in contours_with_children
        ]

        largest_contour = max(parent_circular_contours, key=cv2.contourArea, default=None)
        if largest_contour is None:
            return False, None, image

        # Create the DetectionsAndTime object
        result, detections = detections_and_time.DetectionsAndTime.create(timestamp)
        if not result:
            return False, None, image

        x, y, w, h = cv2.boundingRect(largest_contour)
        bounds = np.array([x, y, x + w, y + h])
        confidence = 1.0  # Confidence for classical CV is often set to a constant value
        label = 0  # Label can be set to a constant or derived from some logic

        # Create a Detection object and append it to detections
        result, detection = detections_and_time.Detection.create(bounds, label, confidence)
        if result:
            detections.append(detection)

        # Annotate the image
        image_annotated = copy.deepcopy(image)
        cv2.rectangle(image_annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            image_annotated,
            "landing-pad",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        return True, detections, image_annotated

    def run(
        self, data: image_and_time.ImageAndTime
    ) -> "tuple[bool, detections_and_time.DetectionsAndTime | None]":
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
