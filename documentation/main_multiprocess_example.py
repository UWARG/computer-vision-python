"""
main process.
"""
import time
import multiprocessing as mp

from documentation.multiprocess_example.add_random import add_random_worker
from documentation.multiprocess_example.concatenator import concatenator_worker
from documentation.multiprocess_example.countup import countup_worker
from utilities import manage_worker


# Play with these numbers to see queue bottlenecks
COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE = 5
ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE = 5


# Command: python -m documentation.main_multiprocess_example
if __name__ == "__main__":
    # Main is managing all worker processes and
    # is responsible for creating supporting interprocess communication
    worker_manager = manage_worker.ManageWorker()

    # mp.Queue has possible race conditions in the same process
    # caused by its implementation (background thread work)
    # so a queue from a SyncManager is used instead
    m = mp.Manager()

    # Queue maxsize should always be >= the larger of producers/consumers count
    # Example: Producers 3, consumers 2, so queue maxsize minimum is 3
    countup_to_add_random_queue = m.Queue(COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE)
    add_random_to_concatenator_queue = m.Queue(ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE)

    # Prepare processes
    # Play with these numbers to see process bottlenecks
    countup_workers = [
        mp.Process(target=countup_worker.countup_worker, args=(
            3, 100, countup_to_add_random_queue, worker_manager
        )),
        mp.Process(target=countup_worker.countup_worker, args=(
            2, 200, countup_to_add_random_queue, worker_manager
        ))
    ]

    add_random_workers = [
        mp.Process(target=add_random_worker.add_random_worker, args=(
            252, 10, 5, countup_to_add_random_queue, add_random_to_concatenator_queue, worker_manager
        )),
        mp.Process(target=add_random_worker.add_random_worker, args=(
            350, 4, 1, countup_to_add_random_queue, add_random_to_concatenator_queue, worker_manager
        )),
    ]

    concatenator_workers = [
        mp.Process(target=concatenator_worker.concatenator_worker, args=(
            "Hello ", " world!", add_random_to_concatenator_queue, worker_manager
        )),
        mp.Process(target=concatenator_worker.concatenator_worker, args=(
            "Example ", " code!", add_random_to_concatenator_queue, worker_manager
        )),
    ]

    # Start worker processes
    for worker in countup_workers:
        worker.start()
    for worker in add_random_workers:
        worker.start()
    for worker in concatenator_workers:
        worker.start()

    # Run for some time and then pause
    time.sleep(2)
    worker_manager.request_pause()
    print("Paused")
    time.sleep(4)
    print("Resumed")
    worker_manager.request_resume()
    time.sleep(2)

    # Stop the processes
    worker_manager.request_exit()

    # Fill and drain queues from END TO START
    manage_worker.ManageWorker.fill_and_drain_queue(
        add_random_to_concatenator_queue, ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE
    )
    manage_worker.ManageWorker.fill_and_drain_queue(
        countup_to_add_random_queue, COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE
    )

    # Clean up worker processes
    for worker in countup_workers:
        worker.join()
    for worker in add_random_workers:
        worker.join()
    for worker in concatenator_workers:
        worker.join()

    # We can reset worker_manager in case we want to reuse it
    # Alternatively, create a new ManageWorker instance
    worker_manager.clear_exit()

    print("Done!")
