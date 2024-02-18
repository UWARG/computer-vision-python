"""
For managing workers.
"""
import multiprocessing as mp


class WorkerManager:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    def __init__(self, workers: "list[mp.Process] | None" = None):
        """
        Constructor creates internal queue and semaphore.
        """
        self.__workers = [] if workers is None else workers

    def create_workers(self, count: int, target, args):
        """
        Create identical workers.

        count: Number of workers.
        target: Function.
        args: Arguments to function.
        """
        for _ in range(0, count):
            worker = mp.Process(target=target, args=args)
            self.__workers.append(worker)

    def concatenate_workers(self, workers: "list[mp.Process]"):
        """
        Add workers.
        """
        self.__workers += workers

    def start_workers(self):
        """
        Start workers.
        """
        for worker in self.__workers:
            worker.start()

    def join_workers(self):
        """
        Join workers.
        """
        for worker in self.__workers:
            worker.join()
