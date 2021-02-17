import threading
from genericPipeline import GenericPipeline


class VideoPipeline(GenericPipeline):
    """
    Class to move video from deckLinkSrc to the targetAcquisiton Class
    """
    def __init__(self):
        super().__init__()
        self.package = []


    def addNewFrame(self, frame):
        """
        Called by the producer to add a new frame to the queue
        Returns
        -------
        None
        """
        self.producerLock.acquire()
        self.package.append(frame)
        self.consumerLock.release()

    def getNewFrame(self):
        """
        Get oldest frame (front of array)
        Returns
        -------
        frame
        """
        self.consumerLock.acquire()
        currentFrame = self.package[0] if len(self.package) > 0 else None
        self.producerLock.release()
        return currentFrame

