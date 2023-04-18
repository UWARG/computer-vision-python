"""
Tests process
"""
import multiprocessing as mp
import queue
import time

from utilities import manage_worker
from modules.video_input import video_input_worker
from modules import frame_and_time


VIDEO_INPUT_WORKER_PERIOD = 1.0
CAMERA = 0


if __name__ == "__main__":
    # Setup
    worker_manager = manage_worker.ManageWorker()

    m = mp.Manager()
    out_queue = m.Queue()

    worker = mp.Process(
        target=video_input_worker.video_input_worker,
        args=(CAMERA, VIDEO_INPUT_WORKER_PERIOD, out_queue, worker_manager)
    )

    # Run
    worker.start()

    time.sleep(3)

    worker_manager.request_exit()

    # Test
    while True:
        try:
            input_data: frame_and_time.FrameAndTime = out_queue.get_nowait()
            assert str(type(input_data)) == "<class \'modules.frame_and_time.FrameAndTime\'>"
            assert input_data.frame is not None

        except queue.Empty:
            break

    # Teardown
    worker.join()

    print("Done!")
