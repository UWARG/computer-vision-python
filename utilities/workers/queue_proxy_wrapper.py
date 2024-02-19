"""
Queue.
"""

import multiprocessing.managers
import queue
import time


class QueueProxyWrapper:
    """
    Wrapper for an underlying queue proxy which also stores maxsize.
    """

    __QUEUE_TIMEOUT = 0.1  # seconds
    __QUEUE_DELAY = 0.1  # seconds

    def __init__(self, mp_manager: multiprocessing.managers.SyncManager, maxsize: int = 0) -> None:
        self.queue = mp_manager.Queue(maxsize)
        self.maxsize = maxsize

    def fill_queue_with_sentinel(self, timeout: float = 0.0) -> None:
        """
        Fills the queue with sentinel (None ).

        timeout: Time waiting before giving up, must be greater than 0 .
        """
        if timeout <= 0.0:
            timeout = self.__QUEUE_TIMEOUT

        try:
            self.queue.put(None, timeout=timeout)
            for _ in range(1, self.maxsize):
                self.queue.put(None, timeout=timeout)
        except queue.Full:
            return

    def drain_queue(self, timeout: float = 0.0) -> None:
        """
        Drains the queue.

        timeout: Time waiting in seconds before giving up, must be greater than 0 .
        """
        if timeout <= 0.0:
            timeout = self.__QUEUE_TIMEOUT

        try:
            self.queue.get(timeout=timeout)
        except queue.Empty:
            return

    def fill_and_drain_queue(self) -> None:
        """
        Fill with sentinel and then drain.
        """
        self.fill_queue_with_sentinel()
        time.sleep(self.__QUEUE_DELAY)
        self.drain_queue()
