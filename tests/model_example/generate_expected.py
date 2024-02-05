"""
Generates expected output using pretrained default model and images.
TODO: PointsAndTime
"""

import cv2
import numpy as np
import ultralytics
import pathlib


# Downloaded from: https://github.com/ultralytics/assets/releases
MODEL_PATH = pathlib.Path("tests", "model_example", "yolov8s_ultralytics_pretrained_default.pt")

BUS_IMAGE_PATH = pathlib.Path("tests", "model_example", "bus.jpg")
ZIDANE_IMAGE_PATH = pathlib.Path("tests", "model_example", "zidane.jpg")

SAVE_PATH = pathlib.Path("tests", "model_example")

BUS_IMAGE_ANNOTATED_PATH = pathlib.Path(SAVE_PATH, "bus_annotated.png")
ZIDANE_IMAGE_ANNOTATED_PATH = pathlib.Path(SAVE_PATH, "zidane_annotated.png")
BUS_BOUNDING_BOX_PATH = pathlib.Path(SAVE_PATH, "bounding_box_bus.txt")
ZIDANE_BOUNDING_BOX_PATH = pathlib.Path(SAVE_PATH, "bounding_box_zidane.txt")


if __name__ == "__main__":
    model = ultralytics.YOLO(MODEL_PATH)
    image_bus = cv2.imread(BUS_IMAGE_PATH)
    image_zidane = cv2.imread(ZIDANE_IMAGE_PATH)

    # ultralytics saves as .jpg , bad for testing reproducibility
    results_bus = model.predict(
            source=image_bus,
            half=True,
            stream=False,
    )

    results_zidane = model.predict(
            source=image_zidane,
            half=True,
            stream=False,
    )

    # Generate image
    image_bus_annotated = results_bus[0].plot(conf=True)
    image_zidane_annotated = results_zidane[0].plot(conf=True)

    # Generate expected
    bounding_box_bus = results_bus[0].boxes.xyxy.detach().cpu().numpy()
    bounding_box_zidane = results_zidane[0].boxes.xyxy.detach().cpu().numpy()

    conf_bus = results_bus[0].boxes.conf.detach().cpu().numpy()
    conf_zidane = results_zidane[0].boxes.conf.detach().cpu().numpy()

    labels_bus = results_bus[0].boxes.cls.detach().cpu().numpy()
    labels_zidane = results_zidane[0].boxes.cls.detach().cpu().numpy()

    predictions_bus = np.insert(bounding_box_bus, 0, [conf_bus, labels_bus], axis=1)
    predictions_zidane = np.insert(bounding_box_zidane, 0, [conf_zidane, labels_zidane], axis=1)

    # Save image
    cv2.imwrite(BUS_IMAGE_ANNOTATED_PATH, image_bus_annotated)
    cv2.imwrite(ZIDANE_IMAGE_ANNOTATED_PATH, image_zidane_annotated)
    
    # Save expected to text file
    np.savetxt(BUS_BOUNDING_BOX_PATH, predictions_bus)
    np.savetxt(ZIDANE_BOUNDING_BOX_PATH, predictions_zidane)

    print("Done!")
