from ..targetAcquisition.targetAcquisition import TargetAcquisition
from ..decklinksrc.decklinksrc import DeckLinkSRC
import threading
import queue


class VideoMediator:

    def __init__(self):
        self.videoPipeline = queue.Queue()
        self.coordinatePipeline = queue.Queue()
        self.coordinates = []
        DeckLinkSRC(self.videoPipeline)
        TargetAcquisition(self.videoPipeline, self.coordinatePipeline)
        mainThread = threading.Thread(target=self._mainThread_)
        mainThread.start()

    def _mainThread_(self):
        self.coordinates.append(self.coordinatePipeline.get())
