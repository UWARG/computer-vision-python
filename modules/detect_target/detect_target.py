"""
Detects objects using the provided model.
"""
import time

import cv2
import ultralytics

from .. import image_and_time
from .. import detections_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class DetectTarget:
    """
    Contains the YOLOv8 model for prediction.
    """
    # Required for logging
    # pylint: disable-next=too-many-arguments
    def __init__(self,
                 device: "str | int",
                 model_path: str,
                 override_full: bool,
                 show_annotations: bool = False,
                 save_name: str = ""):
        """
        device: name of target device to run inference on (i.e. "cpu" or cuda device 0, 1, 2, 3).
        model_path: path to the YOLOv8 model.
        override_full: Force full precision floating point calculations.
        show_annotations: Display annotated images.
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

    # Required for logging
    # pylint: disable-next=too-many-locals
    def run(self,
            data: image_and_time.ImageAndTime) \
        -> "tuple[bool, detections_and_time.DetectionsAndTime | None]":
        """
        Runs object detection on the provided image and returns the detections.

        data: Image with a timestamp.

        Return: Success and the detections.
        """
        start_time = time.time()

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

        # Get Pylance to stop complaining
        assert detections is not None

        for i in range(0, boxes.shape[0]):
            bounds = objects_bounds[i]
            label = int(boxes.cls[i])
            confidence = float(boxes.conf[i])
            result, detection = detections_and_time.Detection.create(bounds, label, confidence)
            if result:
                assert detection is not None
                detections.append(detection)

        stop_time = time.time()

        elapsed_time = stop_time - start_time
      
        for pred in predictions: 
            with open('profiler.txt', 'a') as file:
                speeds = pred.speed
                preprocess_speed = round(speeds['preprocess'], 3)
                inference_speed = round(speeds['inference'], 3)
                postprocess_speed = round(speeds['postprocess'], 3)
                elapsed_time_ms = elapsed_time * 1000
                precision_string = "half" if self.__enable_half_precision else "full"


                file.write(f"{preprocess_speed}, {inference_speed}, {postprocess_speed}, {elapsed_time_ms}, {precision_string}\n")

        # Logging
        if self.__filename_prefix != "":
            filename = self.__filename_prefix + str(self.__counter)

            # Object detections
            with open(filename + ".txt", "w", encoding="utf-8") as file:
                # Use internal string representation
                file.write(repr(detections))

            # Annotated image
            cv2.imwrite(filename + ".png", image_annotated)

            self.__counter += 1

        if self.__show_annotations:
            cv2.imshow("Annotated", image_annotated)

        return True, detections

# pylint: enable=too-few-public-methods
