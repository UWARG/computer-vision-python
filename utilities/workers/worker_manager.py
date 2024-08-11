"""
For managing workers.
"""

import inspect
import multiprocessing as mp

from modules.common.logger import logger
from utilities.workers import worker_controller
from utilities.workers import queue_proxy_wrapper


class WorkerProperties:
    """
    Worker Properties.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        count: int,
        target: "(...) -> object",  # type: ignore
        work_arguments: "tuple",
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
        local_logger: logger.Logger,
    ) -> "tuple[bool, WorkerProperties | None]":
        """
        Creates worker properties.

        count: Number of workers.
        target: Function.
        work_arguments: Arguments for worker internals.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.
        local_logger: Existing logger from process.

        Returns the WorkerProperties object.
        """
        if count <= 0:
            frame = inspect.currentframe()
            local_logger.error(
                "Worker count requested is less than or equal to zero, no workers were created",
                frame,
            )
            return False, None

        return True, WorkerProperties(
            cls.__create_key,
            count,
            target,
            work_arguments,
            input_queues,
            output_queues,
            controller,
        )

    def __init__(
        self,
        class_private_create_key: object,
        count: int,
        target: "(...) -> object",  # type: ignore
        work_arguments: "tuple",
        input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        controller: worker_controller.WorkerController,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is WorkerProperties.__create_key, "Use create() method"

        self.__count = count
        self.__target = target
        self.__work_arguments = work_arguments
        self.__input_queues = input_queues
        self.__output_queues = output_queues
        self.__controller = controller

    def get_worker_arguments(self) -> "tuple":
        """
        Concatenates the worker properties into a tuple.

        Returns the worker properties as a tuple.
        """
        return (
            self.__work_arguments
            + tuple(self.__input_queues)
            + tuple(self.__output_queues)
            + (self.__controller,)
        )

    def get_worker_count(self) -> int:
        """
        Returns the worker count.
        """
        return self.__count

    def get_worker_target(self) -> "(...) -> object":  # type: ignore
        """
        Returns the worker target.
        """
        return self.__target

    def get_input_queues(self) -> "list[queue_proxy_wrapper.QueueProxyWrapper]":
        """
        Returns the input queues.
        """
        return self.__input_queues

    def get_target_name(self) -> str:
        """
        Returns the name of the target.
        """
        return self.__target.__name__


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        worker_properties: WorkerProperties,
        local_logger: logger.Logger,
    ) -> "tuple[bool, WorkerManager | None]":
        """
        Create identical workers and append them to a workers list.

        worker_properties: Worker properties.
        local_logger: Existing logger from process.

        Returns whether the workers were able to be created and the Worker Manager.
        """
        workers = []
        for _ in range(0, worker_properties.get_worker_count()):
            result, worker = WorkerManager.__create_single_worker(
                worker_properties.get_worker_target(),
                worker_properties.get_worker_arguments(),
                local_logger,
            )
            if not result:
                frame = inspect.currentframe()
                local_logger.error("Failed to create worker", frame)
                return False, None

            workers.append(worker)

        return True, WorkerManager(
            cls.__create_key,
            workers,
            worker_properties,
            local_logger,
        )

    def __init__(
        self,
        class_private_create_key: object,
        workers: "list[mp.Process]",
        worker_properties: WorkerProperties,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is WorkerManager.__create_key, "Use create() method"

        self.__workers = workers
        self.__worker_properties = worker_properties
        self.__local_logger = local_logger

    @staticmethod
    def __create_single_worker(target: "(...) -> object", args: "tuple", local_logger: logger.Logger) -> "tuple[bool, mp.Process | None]":  # type: ignore
        """
        Creates a single worker.

        target: Function.
        args: Target function arguments.
        local_logger: Existing logger from process.

        Returns whether a worker was created and the worker.
        """
        try:
            worker = mp.Process(target=target, args=args)
        # Catching all exceptions for library call
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            frame = inspect.currentframe()
            local_logger.error(f"Exception raised while creating a worker: {e}", frame)
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

    def check_and_restart_dead_workers(self) -> bool:
        """
        Check and restart dead workers.

        Returns whether the dead workers were able to be restarted.
        """
        new_workers = []
        for worker in self.__workers:
            if worker.is_alive():
                new_workers.append(worker)
                continue

            # Log dead worker
            frame = inspect.currentframe()
            target_and_worker_name = f"{self.__worker_properties.get_target_name()} {worker.name}"
            self.__local_logger.warning(
                f"Worker died, restarting {target_and_worker_name}",
                frame,
            )

            # Create a new worker
            result, new_worker = WorkerManager.__create_single_worker(
                self.__worker_properties.get_worker_target(),
                self.__worker_properties.get_worker_arguments(),
                self.__local_logger,
            )
            if not result:
                frame = inspect.currentframe()
                self.__local_logger.error(f"Failed to restart {target_and_worker_name}", frame)
                return False

            # Append the new worker
            new_workers.append(new_worker)

        self.__workers = new_workers

        # Draining the preceding queue ensures that the preceding queue data wasn't what
        # caused the worker to fail. Draining the succeeding queues is not needed
        # because a worker that died would not have put bad data into the queue.
        input_queues = self.__worker_properties.get_input_queues()
        for queue in input_queues:
            queue.drain_queue()

        return True
