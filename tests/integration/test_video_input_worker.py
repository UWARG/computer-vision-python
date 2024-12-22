"""
Test worker process.
"""

import multiprocessing as mp
import queue
import time

from modules.common.modules.camera import camera_factory
from modules.common.modules.camera import camera_opencv
from modules.video_input import video_input_worker
from modules import image_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


# Modify these settings as needed
VIDEO_INPUT_WORKER_PERIOD = 1.0
CAMERA = camera_factory.CameraOption.OPENCV
WIDTH = 640
HEIGHT = 480
CONFIG = camera_opencv.ConfigOpenCV(0)
IMAGE_NAME = None  # Not saving any pictures


def main() -> int:
    """
    Main function.
    """
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=video_input_worker.video_input_worker,
        args=(
            CAMERA,
            WIDTH,
            HEIGHT,
            CONFIG,
            IMAGE_NAME,
            VIDEO_INPUT_WORKER_PERIOD,
            out_queue,
            controller,
        ),
    )

    # Run
    worker.start()

    time.sleep(3)

    controller.request_exit()

    # Test
    while True:
        try:
            input_data: image_and_time.ImageAndTime = out_queue.queue.get_nowait()
            assert str(type(input_data)) == "<class 'modules.image_and_time.ImageAndTime'>"
            assert input_data.image is not None

        except queue.Empty:
            break

    # Teardown
    worker.join()

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
