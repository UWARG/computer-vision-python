"""
Tests process
"""
import multiprocessing as mp
import queue
import time

from utilities import manage_worker
from modules.video_input import video_input_worker
from modules import frame_and_time


CAMERA=0


if __name__ == "__main__":
    # Setup
    process_manager = manage_worker.ManageWorker()
    out_queue = mp.Queue()

    worker = mp.Process(
        target=video_input_worker.video_input_worker,
        args=(CAMERA, out_queue, process_manager)
    )

    # Run
    worker.start()

    time.sleep(3)

    process_manager.request_exit()

    # Test
    while True:
        try:
            input_data: frame_and_time.FrameAndTime = out_queue.get_nowait()
            assert str(type(input_data)) == "<class \'modules.frame_and_time.FrameAndTime\'>"
            assert input_data.frame is not None

        except queue.Empty:
            break

    print("Done!")
