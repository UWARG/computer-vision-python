"""
For managing workers.
"""

import inspect
import multiprocessing as mp
from typing import Tuple

from modules.logger import logger
from utilities.workers import worker_controller
from utilities.workers import queue_proxy_wrapper


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    @classmethod
    def create(
        cls,
        count: int,
        target: "(...) -> object",  # type: ignore
        class_args: tuple,
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
    ) -> "list[mp.Process]":
        """
        Create identical workers and append them to a workers list.

        count: Number of workers.
        target: Function.
        class_args: Arguments to function.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns whether the workers were able to be created and the Worker Manager.
        """

        create_logger_result, worker_manager_logger = logger.Logger.create("worker_manager")
        if not create_logger_result:
            print("Error creating worker_manager_logger")
            return False, None

        frame = inspect.currentframe()
        worker_manager_logger.info("worker manager logger initialized", frame)

        if count <= 0:
            return False, None

        args = WorkerManager.create_worker_arguments(
            class_args, input_queues, output_queues, controller
        )

        workers = []
        for _ in range(0, count):
            result, worker = WorkerManager.create_single_worker(target, args, worker_manager_logger)
            if not result:
                frame = inspect.currentframe()
                worker_manager_logger.error("Failed to create worker", frame)
                return False, None

            workers.append(worker)

        return True, WorkerManager(
            workers,
        )

    @staticmethod
    def create_worker_arguments(
        class_args: tuple,
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
    ) -> tuple:
        """
        Creates a tuple containing most arguments for a worker.

        class_args: Class arguments.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns a tuple with the arguments.
        """
        return class_args + tuple(input_queues) + tuple(output_queues) + (controller,)

    @staticmethod
    def create_single_worker(target: "(...) -> object", args: tuple, worker_manager_logger: logger) -> Tuple[bool, mp.Process]: #type: ignore
        """
        Creates a single worker.

        target: Fuction
        args: Worker arguments.

        Returns whether a worker was created and the worker.
        """
        try:
            worker = mp.Process(target=target, args=args)
        except Exception as e: # pylint: disable=broad-exception-caught
            frame = inspect.currentframe()
            worker_manager_logger.error(f"Exception raised while creating a worker: {e}", frame)
            return False, None

        return True, worker

    def __init__(
        self,
        workers: "list[mp.Process]",
    ) -> None:
        """
        Constructor.
        """
        self.__workers = workers

    def concatenate_workers(self, workers: "list[mp.Process]") -> None:
        """
        Add workers.
        """
        self.__workers += workers

    def start_workers(self) -> None:
        """
        Start workers.
        """
        for worker in self.__workers:
            worker.start()

    def join_workers(self) -> None:
        """
        Join workers.
        """
        for worker in self.__workers:
            worker.join()
