"""
Profile detect target using full/half precision.
"""
import multiprocessing as mp
import time
import gc
import pathlib
import yaml
import argparse
import cv2


import numpy as np
import os
import pandas as pd


from modules.detect_target import detect_target
from modules.image_and_time import ImageAndTime






CONFIG_FILE_PATH = pathlib.Path("config.yaml")


GRASS_DATA_DIR = "profiler/profile_data/Grass"
ASPHALT_DATA_DIR = "profiler/profile_data/Asphalt"
FIELD_DATA_DIR = "profiler/profile_data/Field"


THROUGHPUT_TEXT_WORK_COUNT = 50
OVERRIDE_FULL = False
MS_TO_NS_CONV = 1000000




def load_images(dir):
    images = []
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir, filename))
            success, image_with_time = ImageAndTime.create(img)
            if success:
                images.append(image_with_time)
    return images


def profile_detector(detector: detect_target.DetectTarget, images: "list[np.ndarray]") -> dict:
    times_arr = []
    for image in images:
        gc.disable()  # This disables the garbage collector
        start = time.time_ns()
        result, value = detector.run(image)  # Might or might not want to keep the bounding boxes
        end = time.time_ns()
        gc.enable()  # This enables the garbage collector
        if not result:
            pass
            # Handle error
        else:
            times_arr.append(end - start)
   
    if len(times_arr) > 0:
        average = np.nanmean(times_arr) / MS_TO_NS_CONV
        mins = np.nanmin(times_arr) /MS_TO_NS_CONV
        maxs = np.nanmax(times_arr) / MS_TO_NS_CONV
        median = np.median(times_arr) /MS_TO_NS_CONV
    else:
        average, mins, maxs, median = -1,-1,-1,-1


    data = {
        "Average (ms)": average,
        "Min (ms)": mins,
        "Max (ms)": maxs,
        "Median (ms)": median
    }




        # Create and prints DF
    return data


def run_detector(detector_full: detect_target.DetectTarget, detector_half: detect_target.DetectTarget, images: "list[np.ndarray]") -> pd.DataFrame:
    # Initial run just to warm up CUDA
    _ = profile_detector(detector_full, images[:10])
    half_data = profile_detector(detector_half, images)
    full_data = profile_detector(detector_full, images)


    full_df = pd.DataFrame(full_data, index=['full'])
    half_df = pd.DataFrame(half_data, index=['half'])
    return pd.concat([half_df, full_df])


def main() -> int:
    #Configurations
    try:
        with CONFIG_FILE_PATH.open("r", encoding="utf8") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                return -1
    except FileNotFoundError:
        print(f"File not found: {CONFIG_FILE_PATH}")
        return -1
    except IOError as exc:
        print(f"Error when opening file: {exc}")
        return -1
   


    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="option to force cpu")
    args = parser.parse_args()


    DETECT_TARGET_MODEL_PATH = config["detect_target"]["model_path"]
    DETECT_TARGET_DEVICE =  "cpu" if args.cpu else config["detect_target"]["device"]


    #Optional logging parameters
    LOG_DIRECTORY_PATH = config["log_directory_path"]
    DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
    DETECT_TARGET_SAVE_PREFIX = f"{LOG_DIRECTORY_PATH}/{DETECT_TARGET_SAVE_NAME_PREFIX}"


    #Creating detector instances
    detector_half = detect_target.DetectTarget(
        DETECT_TARGET_DEVICE,
        DETECT_TARGET_MODEL_PATH,
        False,
        "" #not logging imgs
    )
    detector_full = detect_target.DetectTarget(
        DETECT_TARGET_DEVICE,
        DETECT_TARGET_MODEL_PATH,
        True, #forces full precision
        "" #not logging imgs
    )


    #Loading images
    grass_images = load_images(GRASS_DATA_DIR)
    asphalt_images = load_images(ASPHALT_DATA_DIR)
    field_images = load_images(FIELD_DATA_DIR)


    #Running detector
    grass_results = run_detector(detector_full, detector_half, grass_images)
    asphalt_results = run_detector(detector_full, detector_half, asphalt_images)
    field_results = run_detector(detector_full, detector_half, field_images)




    #Printing results to console
    print("=================GRASS==================")
    print(grass_results)
    print("=================ASPHALT==================")
    print(asphalt_results)
    print("=================FIELD==================")
    print(field_results)


    #save to csvs
    grass_results.to_csv(f"profiler/results/results_grass.csv")
    asphalt_results.to_csv(f"profiler/results/results_asphalt.csv")
    field_results.to_csv(f"profiler/results/results_field.csv")


if __name__ == "__main__":
    main()







