import threading
import queue
from abc import ABC

class GenericPipeline(ABC):
    """
    Generic Pipeline Class
    """
    def __init__(self):
        self.package = queue.Queue()
        self.consumerLock = threading.Lock()
        self.producerLock = threading.Lock()
        self.consumerLock.acquire()


    def addNewPackage(self, newPackage):
        """
        Called by the producer to update packages
        Returns
        -------
        None
        """
        self.producerLock.acquire()
        self.package.put(newPackage)
        self.consumerLock.release()

    def getNewPackage(self):
        """
        Get the oldest package
        Returns
        -------
        package
        """
        self.consumerLock.acquire()
        currentPackage = self.package.get()
        self.producerLock.release()
        return currentPackage

