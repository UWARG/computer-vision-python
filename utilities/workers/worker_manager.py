"""
For managing workers.
"""

import multiprocessing as mp


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        self.__workers: "list[mp.Process]" = []

    def create_workers(self, count: int, target: "(...) -> object", args: tuple) -> None:  # type: ignore
        """
        Create identical workers.

        count: Number of workers.
        target: Function.
        args: Arguments to function.
        """
        for _ in range(0, count):
            worker = mp.Process(target=target, args=args)
            self.__workers.append(worker)

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
