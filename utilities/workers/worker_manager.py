"""
For managing workers.
"""

import multiprocessing as mp


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    def __init__(self, workers: "list[mp.Process] | None" = None) -> None:
        """
        Constructor creates internal queue and semaphore.
        """
        self.__workers = [] if workers is None else workers

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

    def are_workers_alive(self) -> bool:
        """
        Check if workers are alive.

        Return: True if all workers are alive. False is any 1 worker is not alive.
        
        """
        for worker in self.__workers:
            if not worker.is_alive():
                return False
        return True

    def terminate_workers(self) -> None:
        """
        Terminate workers.
        """
        for worker in self.__workers:
            worker.terminate()
