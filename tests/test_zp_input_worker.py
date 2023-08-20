"""
Test worker process.
"""
import multiprocessing as mp
import queue
import time

from modules.zp_input import zp_input_worker
from modules import message_and_time
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller


PORT = "/dev/ttyS0"
BAUDRATE = 115_200


if __name__ == "__main__":
    # Setup
    controller = worker_controller.WorkerController()

    mp_manager = mp.Manager()
    telemetry_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)
    request_out_queue = queue_proxy_wrapper.QueueProxyWrapper(mp_manager)

    worker = mp.Process(
        target=zp_input_worker.zp_input_worker,
        args=(PORT, BAUDRATE, telemetry_out_queue, request_out_queue, controller),
    )

    # Run
    worker.start()

    time.sleep(3)

    controller.request_exit()

    # Test
    while True:
        try:
            input_data: message_and_time.MessageAndTime = telemetry_out_queue.queue.get_nowait()
            assert str(type(input_data)) == \
                "<class \'modules.message_and_time.MessageAndTime\'>"
            assert input_data.message.header.type == 0

        except queue.Empty:
            break

    while True:
        try:
            input_data: message_and_time.MessageAndTime = request_out_queue.queue.get_nowait()
            assert str(type(input_data)) == \
                "<class \'modules.message_and_time.MessageAndTime\'>"
            assert input_data.message.header.type == 1

        except queue.Empty:
            break

    # Teardown
    worker.join()

    print("Done!")
