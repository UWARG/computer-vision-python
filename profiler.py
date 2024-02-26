import argparse
import multiprocessing as mp
import pathlib
import queue
import cProfile
import time
import numpy as np 
import os
import pandas as pd

import cv2
import yaml

from modules import odometry_and_time
from modules.detect_target import detect_target_worker
from modules.flight_interface import flight_interface_worker
from modules.video_input import video_input_worker
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager



CONFIG_FILE_PATH = pathlib.Path("config.yaml")


def main() -> int:
    """
    copied from airside code main function
    """
    # Open config file
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

    # Parse whether or not to force cpu from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="option to force cpu")
    parser.add_argument("--full", action="store_true", help="option to force full precision")
    args = parser.parse_args()

    try:
        QUEUE_MAX_SIZE = config["queue_max_size"]

        LOG_DIRECTORY_PATH = config["log_directory_path"]

        VIDEO_INPUT_CAMERA_NAME = config["video_input"]["camera_name"]
        VIDEO_INPUT_WORKER_PERIOD = config["video_input"]["worker_period"]
        VIDEO_INPUT_SAVE_NAME_PREFIX = config["video_input"]["save_prefix"]
        VIDEO_INPUT_SAVE_PREFIX = f"{LOG_DIRECTORY_PATH}/{VIDEO_INPUT_SAVE_NAME_PREFIX}"

        DETECT_TARGET_WORKER_COUNT = config["detect_target"]["worker_count"]
        DETECT_TARGET_DEVICE =  "cpu" if args.cpu else config["detect_target"]["device"]
        DETECT_TARGET_MODEL_PATH = config["detect_target"]["model_path"]
        DETECT_TARGET_OVERRIDE_FULL_PRECISION = args.full #note: if not set, defaults to False (with profiler implementation)
        DETECT_TARGET_SAVE_NAME_PREFIX = config["detect_target"]["save_prefix"]
        DETECT_TARGET_SAVE_PREFIX = f"{LOG_DIRECTORY_PATH}/{DETECT_TARGET_SAVE_NAME_PREFIX}"
        PROFILING_LENGTH = config["profiling_length"]  # 300 seconds = 5 minutes

    except KeyError:
        print("Config key(s) not found")
        return -1

    pathlib.Path(LOG_DIRECTORY_PATH).mkdir(exist_ok=True)

    # Setup
    if os.path.exists('profiler.txt'):
    # Delete the contents of the profiler.txt file
        open('profiler.txt', 'w').close()
        print("Contents of profiler.txt deleted")

    with open('profiler.txt', 'w') as file:
        file.write("preprocess, inference, postprocess, elapsed_time, half/full precision\n")

    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    video_input_to_detect_target_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    detect_target_to_main_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        QUEUE_MAX_SIZE,
    )
    video_input_manager = worker_manager.WorkerManager()
    video_input_manager.create_workers(
        1,
        video_input_worker.video_input_worker,
        (
            VIDEO_INPUT_CAMERA_NAME,
            VIDEO_INPUT_WORKER_PERIOD,
            VIDEO_INPUT_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            controller,
        ),
    )

    detect_target_manager = worker_manager.WorkerManager()
    detect_target_manager.create_workers(
        DETECT_TARGET_WORKER_COUNT,
        detect_target_worker.detect_target_worker,
        (
            DETECT_TARGET_DEVICE,
            DETECT_TARGET_MODEL_PATH,
            DETECT_TARGET_OVERRIDE_FULL_PRECISION,
            DETECT_TARGET_SAVE_PREFIX,
            video_input_to_detect_target_queue,
            detect_target_to_main_queue,
            controller,
        ),
    )


    # Run
    video_input_manager.start_workers()
    detect_target_manager.start_workers()

    start_time = time.time()

    while True:
        try:
            if time.time() - start_time > PROFILING_LENGTH:  # 300 seconds = 5 minutes
                break
            image = detect_target_to_main_queue.queue.get_nowait()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            image = None


    controller.request_exit()

    video_input_to_detect_target_queue.fill_and_drain_queue()
    detect_target_to_main_queue.fill_and_drain_queue()

    video_input_manager.join_workers()
    detect_target_manager.join_workers()


    #====PROFILING CODE FOR METRIC CALCULATIONS=====
    # Read data from the text file
    timing_data = [] #stores raw timing data (float)
    column_names = [] #stores col names (str)
    header_row = True  # Flag to identify row of column names 


    with open('profiler.txt', 'r') as file:
        for line in file:
            if header_row:
                header_row = False
                column_names = line.strip().split(',')
                continue  # Skip processing the first row
            
            row = line.strip().split(',')
            try:
            # Convert all elements except the last one to float and append to data
                row_except_last = [float(value) for value in row[:-1]]
                timing_data.append(row_except_last)
            except ValueError:
                print(f"Skipping invalid data: {line.strip()}")

    # Convert the data into a numpy array for metric calculations
    data_array = np.array(timing_data)


    # Check if the data array is empty
    if data_array.size == 0:
        print("No data found.")
    else:
        # Calculates metrics (skips first row of data which is skewed - see profiler.txt)
        averages = np.nanmean(data_array[1:], axis=0)
        mins = np.nanmin(data_array[1:], axis=0)
        maxs = np.nanmax(data_array[1:], axis=0)  
        medians = np.median(data_array[1:], axis=0)
        initial = data_array[0]
    

        # Create and prints DF
        df = pd.DataFrame({'Average (ms)': averages, 'Min (ms)': mins, 'Max (ms)': maxs, 'Median (ms)': medians, 'Initial Pred (ms)': initial}, index=column_names[:-1])
        print(f"Profiling results for {'full' if DETECT_TARGET_OVERRIDE_FULL_PRECISION else 'half'}:")
        print(df)

        


    return 0


if __name__ == "__main__":
    result_run = main()
    if result_run < 0:
        print(f"ERROR: Status code: {result_run}")

    print("Done!")
