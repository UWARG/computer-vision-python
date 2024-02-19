"""
For controlling workers.
"""

import multiprocessing as mp
import time


class WorkerController:
    """
    For interprocess communication from main to worker.
    Contains exit and pause requests.
    """

    __QUEUE_DELAY = 0.1  # seconds

    def __init__(self) -> None:
        """
        Constructor creates internal queue and semaphore.
        """
        self.__pause = mp.BoundedSemaphore(1)
        self.__is_paused = False
        self.__exit_queue = mp.Queue(1)

    def request_pause(self) -> None:
        """
        Requests worker processes to pause.
        """
        if not self.__is_paused:
            self.__pause.acquire()
            self.__is_paused = True

    def request_resume(self) -> None:
        """
        Requests worker processes to resume.
        """
        if self.__is_paused:
            self.__pause.release()
            self.__is_paused = False

    def check_pause(self) -> None:
        """
        Blocks worker if main has requested it to pause, otherwise continues.
        """
        self.__pause.acquire()
        self.__pause.release()

    def request_exit(self) -> None:
        """
        Requests worker processes to exit.
        Does nothing if already requested.
        """
        time.sleep(self.__QUEUE_DELAY)
        if self.__exit_queue.empty():
            self.__exit_queue.put(None)

    def clear_exit(self) -> None:
        """
        Clears the exit request condition.
        Does nothing if already cleared.
        """
        time.sleep(self.__QUEUE_DELAY)
        if not self.__exit_queue.empty():
            _ = self.__exit_queue.get()

    def is_exit_requested(self) -> bool:
        """
        Returns whether main has requested the worker process to exit.
        There is a race condition, but it's fine because the worker process
        will do at most 1 additional loop.
        """
        return not self.__exit_queue.empty()
