"""
For model training
"""

import ultralytics


if __name__ == "__main__":

    # Load model
    # Use nano for now
    model = ultralytics.YOLO("yolov8n.yaml")

    # Train
    # Configurations: https://docs.ultralytics.com/usage/cfg/
    model.train(
        data="training/2023_pad.yaml",
        imgsz=720,
        save=True,
        save_period=10,
        device=0,
        workers=4,
        verbose=True,
        resume=False,  # Change to True if interrupted
    )

    print("Done!")
