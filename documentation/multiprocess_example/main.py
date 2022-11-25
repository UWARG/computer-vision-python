"""
main process.
"""
import multiprocessing as mp
import time

import add_random_worker
import concatenator_worker
import countup_worker
# Import required beyond the current directory
# pylint: disable=relative-beyond-top-level
from ...utilities.manage_worker import ManageWorker
# pylint: enable=relative-beyond-top-level


# Play with these numbers to see queue bottlenecks
COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE = 5
ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE = 5


if __name__ == "__main__":
    # Main is managing all worker processes and
    # is responsible for creating supporting interprocess communication
    main_control = ManageWorker()
    # Queue maxsize should always be >= the larger of producers/consumers count
    # Example: Producers 3, consumers 2, so queue maxsize minimum is 3
    countup_to_add_random_queue = mp.Queue(COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE)
    add_random_to_concatenator_queue = mp.Queue(ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE)

    # Prepare processes
    # Play with these numbers to see process bottlenecks
    countup_workers = [
        mp.Process(target=countup_worker.countup_worker, args=(
            3, 100, countup_to_add_random_queue, main_control
        )),
        mp.Process(target=countup_worker.countup_worker, args=(
            2, 200, countup_to_add_random_queue, main_control
        ))
    ]

    add_random_workers = [
        mp.Process(target=add_random_worker.add_random_worker, args=(
            252, 10, 5, countup_to_add_random_queue, add_random_to_concatenator_queue, main_control
        )),
        mp.Process(target=add_random_worker.add_random_worker, args=(
            350, 4, 1, countup_to_add_random_queue, add_random_to_concatenator_queue, main_control
        )),
    ]

    concatenator_workers = [
        mp.Process(target=concatenator_worker.concatenator_worker, args=(
            "Hello ", " world!", add_random_to_concatenator_queue, main_control
        )),
        mp.Process(target=add_random_worker.add_random_worker, args=(
            "Example ", " code!", add_random_to_concatenator_queue, main_control
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
    main_control.request_pause()
    print("Paused")
    time.sleep(4)
    print("Resumed")
    main_control.request_resume()
    time.sleep(2)

    # Stop the processes
    main_control.request_exit()

    # Fill and drain queues from end to start
    ManageWorker.fill_and_drain_queue(add_random_to_concatenator_queue)
    ManageWorker.fill_and_drain_queue(countup_to_add_random_queue)

    # We can reset the stop in case we want to reuse it
    # Alternatively, create a new ManageWorker instance
    main_control.clear_exit()

    print("Done!")
