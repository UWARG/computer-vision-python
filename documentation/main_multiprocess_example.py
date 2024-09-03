"""
Main process. To run:
```
python -m documentation.main_multiprocess_example
```
"""

import multiprocessing as mp
import time

from documentation.multiprocess_example.add_random import add_random_worker
from documentation.multiprocess_example.concatenator import concatenator_worker
from documentation.multiprocess_example.countup import countup_worker
from modules.common.logger.modules import logger
from modules.common.logger.modules import logger_setup_main
from modules.common.logger.read_yaml.modules import read_yaml
from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller
from utilities.workers import worker_manager


# Play with these numbers to see queue bottlenecks
COUNTUP_TO_ADD_RANDOM_QUEUE_MAX_SIZE = 5
ADD_RANDOM_TO_CONCATENATOR_QUEUE_MAX_SIZE = 5

# Play with these numbers to see process bottlenecks
COUNTUP_WORKER_COUNT = 2
ADD_RANDOM_WORKER_COUNT = 2
CONCATENATOR_WORKER_COUNT = 2


# main() is required for early return
def main() -> int:
    """
    Main function.
    """
    # Configuration settings
    result, config = read_yaml.open_config(logger.CONFIG_FILE_PATH)
    if not result:
        print("ERROR: Failed to load configuration file")
        return -1

    # Get Pylance to stop complaining
    assert config is not None

    # Setup main logger
    result, main_logger, _ = logger_setup_main.setup_main_logger(config)
    if not result:
        print("ERROR: Failed to create main logger")
        return -1

    # Get Pylance to stop complaining
    assert main_logger is not None

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

    # Worker properties
    result, countup_worker_properties = worker_manager.WorkerProperties.create(
        count=COUNTUP_WORKER_COUNT,
        target=countup_worker.countup_worker,
        work_arguments=(
            3,
            100,
        ),
        input_queues=[],
        output_queues=[countup_to_add_random_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create arguments for Countup")
        return -1

    # Get Pylance to stop complaining
    assert countup_worker_properties is not None

    result, add_random_worker_properties = worker_manager.WorkerProperties.create(
        count=ADD_RANDOM_WORKER_COUNT,
        target=add_random_worker.add_random_worker,
        work_arguments=(
            252,
            10,
            5,
        ),
        input_queues=[countup_to_add_random_queue],
        output_queues=[add_random_to_concatenator_queue],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create arguments for Add Random")
        return -1

    # Get Pylance to stop complaining
    assert add_random_worker_properties is not None

    result, concatenator_worker_properties = worker_manager.WorkerProperties.create(
        count=CONCATENATOR_WORKER_COUNT,
        target=concatenator_worker.concatenator_worker,
        work_arguments=(
            "Hello ",
            " world!",
        ),
        input_queues=[add_random_to_concatenator_queue],
        output_queues=[],
        controller=controller,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create arguments for Concatenator")
        return -1

    # Get Pylance to stop complaining
    assert concatenator_worker_properties is not None

    # Prepare processes
    # Data path: countup_worker to add_random_worker to concatenator_workers
    worker_managers = []

    result, countup_manager = worker_manager.WorkerManager.create(
        worker_properties=countup_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create manager for Countup")
        return -1

    # Get Pylance to stop complaining
    assert countup_manager is not None

    worker_managers.append(countup_manager)

    result, add_random_manager = worker_manager.WorkerManager.create(
        worker_properties=add_random_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create manager for Add Random")
        return -1

    # Get Pylance to stop complaining
    assert add_random_manager is not None

    worker_managers.append(add_random_manager)

    result, concatenator_manager = worker_manager.WorkerManager.create(
        worker_properties=concatenator_worker_properties,
        local_logger=main_logger,
    )
    if not result:
        print("Failed to create manager for Concatenator")
        return -1

    # Get Pylance to stop complaining
    assert concatenator_manager is not None

    worker_managers.append(concatenator_manager)

    # Start worker processes
    for manager in worker_managers:
        manager.start_workers()

    main_logger.info("Started", True)

    # Run for some time and then pause
    time.sleep(2)
    controller.request_pause()

    main_logger.info("Paused", True)

    time.sleep(4)
    controller.request_resume()
    main_logger.info("Resumed", True)

    time.sleep(2)

    # Stop the processes
    controller.request_exit()

    main_logger.info("Requested exit", True)

    # Fill and drain queues from END TO START
    countup_to_add_random_queue.fill_and_drain_queue()
    add_random_to_concatenator_queue.fill_and_drain_queue()

    main_logger.info("Queues cleared", True)

    # Clean up worker processes
    for manager in worker_managers:
        manager.join_workers()

    main_logger.info("Stopped", True)

    # We can reset controller in case we want to reuse it
    # Alternatively, create a new WorkerController instance
    controller.clear_exit()

    return 0


# Main guard is only used to call main()
if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
