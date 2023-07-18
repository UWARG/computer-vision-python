"""
Detects objects using the provided model
"""
import time

import cv2
import numpy as np  # TODO: Remove
import ultralytics

from .. import frame_and_time
from .. import detections_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class DetectTarget:
    """
    Contains the YOLOv8 model for prediction
    """
    def __init__(self, device: "str | int", model_path: str, save_name: str = ""):
        self.__device = device
        self.__model = ultralytics.YOLO(model_path)
        self.__counter = 0
        self.__filename_prefix = ""
        if save_name != "":
            self.__filename_prefix = save_name + "_" + str(int(time.time())) + "_"

    def run(self, data: frame_and_time.FrameAndTime) -> "tuple[bool, np.ndarray | None]":
        """
        Returns annotated image
        TODO: Change to DetectionsAndTime
        """
        image = data.frame
        predictions = self.__model.predict(
            source=image,
            half=True,
            device=self.__device,
            stream=False,
        )

        if len(predictions) == 0:
            return False, None

        # TODO: Change this to DetectionsAndTime for image and telemetry merge for 2024
        image_annotated = predictions[0].plot(conf=True)

        # Processing object detection
        boxes = predictions[0].boxes
        if boxes.shape[0] == 0:
            return False, None

        objects_bounds = boxes.xyxy.detach().cpu().numpy()
        detections = detections_and_time.DetectionsAndTime(data.timestamp)
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

        # TODO: Change this to DetectionsAndTime
        return True, image_annotated

# pylint: enable=too-few-public-methods
