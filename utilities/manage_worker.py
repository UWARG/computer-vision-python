"""
For interprocess communication from main to worker.
"""
import multiprocessing as mp
import queue
import time


QUEUE_DELAY = 1  # seconds


class ManageWorker:
    """
    Contains exit and pause requests.
    """
    def __init__(self):
        """
        Constructor creates internal queue and semaphore.
        """
        self.__pause = mp.BoundedSemaphore(1)
        self.__exit_queue = mp.Queue(1)

    def request_pause(self) -> None:
        """
        Requests worker processes to pause.
        Should only be called while workers are resumed.
        """
        self.__pause.acquire()

    def request_resume(self) -> None:
        """
        Requests worker processes to resume.
        Should only be called while workers are paused.
        """
        self.__pause.release()

    def check_pause(self) -> None:
        """
        Blocks worker if main has requested it to pause, otherwise continues.
        """
        self.__pause.acquire()
        self.__pause.release()

    def request_exit(self) -> None:
        """
        Requests worker processes to exit.
        """
        self.__exit_queue.put(None)

    def clear_exit(self) -> None:
        """
        Clears the exit request condition.
        """
        if not self.__exit_queue.empty():
            self.__exit_queue.get()

    def is_exit_requested(self) -> bool:
        """
        Returns whether main has requested the worker process to exit.
        There is a race condition, but it's fine
        because the worker process will do at most 1 additional loop.
        """
        return not self.__exit_queue.empty()

    @staticmethod
    def fill_and_drain_queue(worker_queue: queue.Queue, worker_queue_max_size: int=1) -> None:
        """
        In case the processes are stuck on a queue.

        The maximum number of processes stuck on a single in or out queue
        is the number of running processes (obviously).
        They may all be waiting for data (stuck on empty queue),
        so we push in that number of elements.
        They may all be trying to output data (stuck on full queue),
        so we pop that number of elements.
        We don't really care about the data any more because the whole system is halting.

        This assumes that the queue maxsize is >= than the number of producers or consumers.
        """
        for _ in range(0, worker_queue_max_size):
            try:
                worker_queue.put_nowait(None)

            except queue.Full:
                break

        # Let other processes run
        time.sleep(QUEUE_DELAY)

        for _ in range(0, worker_queue_max_size):
            try:
                worker_queue.get_nowait()

            except queue.Empty:
                break
