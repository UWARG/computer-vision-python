"""
Main process.
"""
import multiprocessing as mp
import time

from documentation.multiprocess_example.add_random import add_random_worker
from documentation.multiprocess_example.concatenator import concatenator_worker
from documentation.multiprocess_example.countup import countup_worker
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager


# Play with these numbers to see queue bottlenecks
COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE = 5
ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE = 5


# Command: python -m documentation.main_multiprocess_example
if __name__ == "__main__":
    # Main is managing all worker processes and is responsible
    # for creating supporting interprocess communication
    controller = worker_controller.WorkerController()

    # mp.Queue has possible race conditions in the same process
    # caused by its implementation (background thread work)
    # so a queue from a SyncManager is used instead
    # See 2nd note: https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues
    mp_manager = mp.Manager()

    # Queue maxsize should always be >= the larger of producers/consumers count
    # Example: Producers 3, consumers 2, so queue maxsize minimum is 3
    countup_to_add_random_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE,
    )
    add_random_to_concatenator_queue = queue_proxy_wrapper.QueueProxyWrapper(
        mp_manager,
        ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE,
    )

    # Prepare processes
    # Data path: countup_worker to add_random_worker to concatenator_workers
    # Play with these numbers to see process bottlenecks
    countup_workers = [
        mp.Process(
            target=countup_worker.countup_worker,
            args=(
                3,
                100,
                countup_to_add_random_queue,
                controller,
            ),
        ),
        mp.Process(
            target=countup_worker.countup_worker,
            args=(
                2,
                200,
                countup_to_add_random_queue,
                controller,
            ),
        ),
    ]
    countup_manager = worker_manager.WorkerManager(countup_workers)

    add_random_workers = [
        mp.Process(
            target=add_random_worker.add_random_worker,
            args=(
                252,
                10,
                5,
                countup_to_add_random_queue,
                add_random_to_concatenator_queue,
                controller,
            ),
        ),
        mp.Process(
            target=add_random_worker.add_random_worker,
            args=(
                350,
                4,
                1,
                countup_to_add_random_queue,
                add_random_to_concatenator_queue,
                controller,
            ),
        ),
    ]
    add_random_manager = worker_manager.WorkerManager(add_random_workers)

    concatenator_workers = [
        mp.Process(
            target=concatenator_worker.concatenator_worker,
            args=(
                "Hello ",
                " world!",
                add_random_to_concatenator_queue,
                controller,
            ),
        ),
        mp.Process(
            target=concatenator_worker.concatenator_worker,
            args=(
                "Example ",
                " code!",
                add_random_to_concatenator_queue,
                controller,
            ),
        ),
    ]
    concatenator_manager = worker_manager.WorkerManager(concatenator_workers)

    # Start worker processes
    countup_manager.start_workers()
    add_random_manager.start_workers()
    concatenator_manager.start_workers()

    # Run for some time and then pause
    time.sleep(2)
    controller.request_pause()
    print("Paused")
    time.sleep(4)
    print("Resumed")
    controller.request_resume()
    time.sleep(2)

    # Stop the processes
    controller.request_exit()

    # Fill and drain queues from END TO START
    countup_to_add_random_queue.fill_and_drain_queue()
    add_random_to_concatenator_queue.fill_and_drain_queue()

    # Clean up worker processes
    countup_manager.join_workers()
    add_random_manager.join_workers()
    concatenator_manager.join_workers()

    # We can reset controller in case we want to reuse it
    # Alternatively, create a new WorkerController instance
    controller.clear_exit()

    print("Done!")
