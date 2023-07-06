"""
Tests process
"""
import multiprocessing as mp
import queue
import time

from modules.video_input import video_input_worker
from modules import frame_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


VIDEO_INPUT_WORKER_PERIOD = 1.0
CAMERA = 0


if __name__ == "__main__":
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()

    out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=video_input_worker.video_input_worker,
        args=(CAMERA, VIDEO_INPUT_WORKER_PERIOD, "", out_queue, controller),
    )

    # Run
    worker.start()

    time.sleep(3)

    controller.request_exit()

    # Test
    while True:
        try:
            input_data: frame_and_time.FrameAndTime = out_queue.queue.get_nowait()
            assert str(type(input_data)) == "<class \'modules.frame_and_time.FrameAndTime\'>"
            assert input_data.frame is not None

        except queue.Empty:
            break

    # Teardown
    worker.join()

    print("Done!")
