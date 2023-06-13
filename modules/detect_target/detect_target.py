"""
Detects objects using the provided model
"""

import numpy as np  # TODO: Remove
import ultralytics

from .. import frame_and_time


# This is just an interface
# pylint: disable=too-few-public-methods
class DetectTarget:
    """
    Contains the YOLOv8 model for prediction
    """

    def __init__(self, model_path: str):
        self.model = ultralytics.YOLO(model_path)

    def run(self, data: frame_and_time.FrameAndTime) -> "tuple[bool, np.ndarray | None]":
        """
        Returns annotated image
        TODO: Change to PointsAndTime
        """
        image = data.frame
        predictions = self.model.predict(
            source=image,
            half=True,
            device=0,
            stream=False)

        if len(predictions) == 0:
            return False, None

        # TODO: Change this to image points for image and telemetry merge for 2024
        # (bounding box conversion code required)
        image_annotated = predictions[0].plot(conf=True)

        # TODO: Change this to PointsAndTime
        return True, image_annotated

# pylint: enable=too-few-public-methods
