"""
For managing workers.
"""

import inspect
import multiprocessing as mp

from modules.logger import logger
from utilities.workers import worker_controller
from utilities.workers import queue_proxy_wrapper


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        count: int,
        target: "(...) -> object",  # type: ignore
        class_args: "tuple",
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
    ) -> "tuple[bool, WorkerManager]":
        """
        Create identical workers and append them to a workers list.

        count: Number of workers.
        target: Function.
        class_args: Arguments for worker internals.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns whether the workers were able to be created and the Worker Manager.
        """
        result, worker_manager_logger = logger.Logger.create("worker_manager")
        if not result:
            print("Error creating worker manager logger")
            return False, None

        frame = inspect.currentframe()
        worker_manager_logger.info("worker manager logger initialized", frame)

        if count <= 0:
            frame = inspect.currentframe()
            worker_manager_logger.error(
                "Worker count requested is less than or equal to zero, no workers were created",
                frame,
            )
            return False, None

        args = WorkerManager.__create_worker_arguments(
            class_args, input_queues, output_queues, controller
        )

        workers = []
        for _ in range(0, count):
            result, worker = WorkerManager.__create_single_worker(
                target,
                args,
                worker_manager_logger,
            )
            if not result:
                frame = inspect.currentframe()
                worker_manager_logger.error("Failed to create worker", frame)
                return False, None

            workers.append(worker)

        return True, WorkerManager(
            cls.__create_key,
            workers,
        )

    def __init__(
        self,
        class_private_create_key: object,
        workers: "list[mp.Process]",
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is WorkerManager.__create_key, "Use create() method"

        self.__workers = workers

    @staticmethod
    def __create_worker_arguments(
        class_args: "tuple",
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
    ) -> tuple:
        """
        Creates a tuple containing most arguments for a worker.

        class_args: Arguments for worker internals.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns a tuple with the arguments.
        """
        return class_args + tuple(input_queues) + tuple(output_queues) + (controller,)

    @staticmethod
    def __create_single_worker(target: "(...) -> object", args: "tuple", worker_manager_logger: logger) -> "tuple[bool, mp.Process]":  # type: ignore
        """
        Creates a single worker.

        target: Function.
        args: Target function arguments.
        worker_manager_logger: Logger for the Worker Manager.

        Returns whether a worker was created and the worker.
        """
        try:
            worker = mp.Process(target=target, args=args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            frame = inspect.currentframe()
            worker_manager_logger.error(f"Exception raised while creating a worker: {e}", frame)
            return False, None

        return True, worker

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
