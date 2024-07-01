"""
For managing workers.
"""

import multiprocessing as mp
import types

from typing import Tuple
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
        if count == 0:
            return False, None

        args = WorkerManager.create_worker_arguments(
            class_args, input_queues, output_queues, controller
        )

        workers = []
        for _ in range(0, count):
            result, worker = WorkerManager.create_single_worker(target, args)
            if not result:
                return False, None

            workers.append(worker)

        return True, WorkerManager(
            workers,
            # target, class_args, input_queues, output_queues, controller
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
    def create_single_worker(target: types.FunctionType, args: tuple) -> Tuple[bool, mp.Process]:
        """
        Creates a single worker.

        target: Fuction
        args: Worker arguments.

        Returns whether a worker was created and the worker.
        """
        try:
            worker = mp.Process(target=target, args=args)
        except:
            return False, None

        return True, worker

    def __init__(
        self,
        workers: "list[mp.Process]",
        # target: "(...) -> object", #type: ignore
        # class_args: tuple,
        # input_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        # output_queues: "list[queue_proxy_wrapper.QueueProxyWrapper]",
        # controller: worker_controller.WorkerController,
    ) -> None:
        """
        Constructor.
        """
        self.__workers = workers
        # self.__target = target
        # self.__class_args = class_args
        # self.__input_queues = input_queues
        # self.__output_queues = output_queues
        # self.__controller = controller

    # def check_and_restart_dead_workers(self) -> None:
    #     """
    #     TODO
    #     """
    #     for worker in self.__workers:
    #         if not worker.is_alive():
    #             #TODO: Implement restarting code.
    #             pass

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
