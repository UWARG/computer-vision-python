"""
Detects objects using the provided model.
"""
import time

import cv2
import numpy as np  # TODO: Remove
import ultralytics

from .. import image_and_time
from .. import detections_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class DetectTarget:
    """
    Contains the YOLOv8 model for prediction.
    """
    def __init__(self, device: "str | int", model_path: str, override_full: bool, show_annotations: bool = False, save_name: str = ""):
        """
        device: name of target device to run inference on (i.e. "cpu" or cuda device 0, 1, 2, 3).
        model_path: path to the YOLOv8 model.
        override_full: Force full precision floating point calculations.
        save_name: filename prefix for logging detections and annotated images.
        """
        self.__device = device
        self.__model = ultralytics.YOLO(model_path)
        self.__counter = 0
        self.__enable_half_precision = False if self.__device == "cpu" else True
        self.__show_annotations = show_annotations
        if override_full:
            self.__enable_half_precision = False
        self.__filename_prefix = ""
        if save_name != "":
            self.__filename_prefix = save_name + "_" + str(int(time.time())) + "_"

    def run(self, data: image_and_time.ImageAndTime) -> "tuple[bool, detections_and_time.DetectionsAndTime | None]":
        """
        Returns annotated image.
        """
        print(self.__enable_half_precision)
        image = data.image
        predictions = self.__model.predict(
            source=image,
            half=self.__enable_half_precision,
            device=self.__device,
            stream=False,
        )

        if len(predictions) == 0:
            return False, None

        image_annotated = predictions[0].plot(conf=True)

        # Processing object detection
        boxes = predictions[0].boxes
        if boxes.shape[0] == 0:
            return False, None

        # Make a copy of bounding boxes in CPU space
        objects_bounds = boxes.xyxy.detach().cpu().numpy()
        result, detections = detections_and_time.DetectionsAndTime.create(data.timestamp)
        if not result:
            return False, None

        assert detections is not None

        for i in range(0, boxes.shape[0]):
            bounds = objects_bounds[i]
            label = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            result, detection = detections_and_time.Detection.create(bounds, label, confidence)
            if result:
                assert detection is not None
                detections.append(detection)

        # Logging
        if self.__filename_prefix != "":
            filename = self.__filename_prefix + str(self.__counter)

            # Object detections
            with open(filename + ".txt", "w") as file:
                # Use internal string representation
                file.write(repr(detections))

            # Annotated image
            cv2.imwrite(filename + ".png", image_annotated)

            self.__counter += 1

        if self.__show_annotations:
            cv2.imshow("Annotated", image_annotated)

        return True, detections

# pylint: enable=too-few-public-methods
