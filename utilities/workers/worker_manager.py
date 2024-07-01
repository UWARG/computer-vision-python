"""
For managing workers.
"""

import multiprocessing as mp
from typing import Tuple
from utilities.workers import worker_controller

class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    def __init__(self, workers: "list[mp.Process]", target, class_args, input_queues, output_queues, controller) -> None:
        """
        Constructor creates internal queue and semaphore.
        """
        # self, workers: "list[mp.Process] | None" = None
        # self.__workers = [] if workers is None else workers
        self.__workers = workers
        self.__target = target
        self.__class_args = class_args
        self.__input_queues = input_queues
        self.__output_queues = output_queues
        self.__controller = controller

    @staticmethod
    def create_worker_arguments(class_args, input_queues, output_queues, controller) -> tuple:
        """
        Creates a tuple containing all arguments for a worker.

        class_args: Class arguments.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns a tuple with all arguments.
        """
        args = (class_args + tuple(input_queues) + tuple(output_queues) + (controller,))
        return args

    # Potentially put all of these parameters into its own class
    @staticmethod
    def create_worker(target, class_args, input_queues, output_queues, controller) -> Tuple[bool, mp.Process]:
        """
        Creates a worker.

        target: Fuction
        class_args: Class arguments.
        input_queues: Input queues.
        output_queues: Output queues.
        controller: Worker controller.

        Returns a tuple with all arguments.
        """
        args = WorkerManager.create_worker_arguments(class_args, input_queues, output_queues, controller)
        try:
            print(args)
            worker = mp.Process(target=target, args=args)
        except:
            return False, None

        return True, worker

    @classmethod
    def create(cls, count: int, target: "(...) -> object", class_args: tuple, input_queues: list, output_queues: list, controller: worker_controller.WorkerController) -> "list[mp.Process]":  # type: ignore
        """
        Create identical workers and append them to a workers list.

        count: Number of workers.
        target: Function.
        class_args: Arguments to function.
        """
        if count == 0:
            return False, None

        workers = []
        for _ in range(0, count):
            result, worker = WorkerManager.create_worker(target, class_args, input_queues, output_queues, controller)
            if not result:
                return False, None

            workers.append(worker)

        return True, WorkerManager(workers, target, class_args, input_queues, output_queues, controller)

    def check_and_restart_dead_workers(self) -> None:
        """
        TODO
        """
        for worker in self.__workers:
            if not worker.is_alive():
                # Do the needful
                pass

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
