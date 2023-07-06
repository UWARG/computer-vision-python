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

    def __init__(self, mp_manager: multiprocessing.managers.SyncManager, maxsize: int = 0):
        self.queue = mp_manager.Queue(maxsize)
        self.maxsize = maxsize

    def fill_queue_with_sentinel(self):
        """
        Fills the queue with sentinel (None ).
        """
        self.queue.put(None)
        for _ in range(1, self.maxsize):
            self.queue.put(None)

    def drain_queue(self, timeout: float = 0.0):
        """
        Drains the queue.

        timeout: Time waiting before giving up, must be greater than 0.
        """
        if timeout <= 0.0:
            timeout = self.__QUEUE_TIMEOUT

        try:
            self.queue.get(timeout=timeout)
        except queue.Empty:
            pass

    def fill_and_drain_queue(self):
        """
        Fill with sentinel and then drain.
        """
        self.fill_queue_with_sentinel()
        time.sleep(self.__QUEUE_DELAY)
        self.drain_queue()
