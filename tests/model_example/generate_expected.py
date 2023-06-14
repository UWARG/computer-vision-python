"""
Generates expected output using pretrained default model and images
TODO: PointsAndTime
"""

import cv2
import ultralytics


# Downloaded from: https://github.com/ultralytics/assets/releases
MODEL_PATH = "tests/model_example/yolov8s_ultralytics_pretrained_default.pt"


if __name__ == "__main__":
    model = ultralytics.YOLO(MODEL_PATH)
    image_bus = cv2.imread("tests/model_example/bus.jpg")
    image_zidane = cv2.imread("tests/model_example/zidane.jpg")

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

    # Save image
    cv2.imwrite("tests/model_example/bus_annotated.png", image_bus_annotated)
    cv2.imwrite("tests/model_example/zidane_annotated.png", image_zidane_annotated)

    print("Done!")
