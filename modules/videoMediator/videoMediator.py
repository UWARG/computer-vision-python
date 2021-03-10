from ..targetAcquisition.targetAcquisition import TargetAcquisition
from ..decklinksrc.decklinksrc import DeckLinkSRC
from ...pipelines.genericPipeline import GenericPipeline
import threading


class VideoMediator:

    def __init__(self):
        self.videoPipeline = GenericPipeline()
        self.coordinatePipeline = GenericPipeline()
        self.coordinates = []
        DeckLinkSRC(self.videoPipeline)
        TargetAcquisition(self.videoPipeline, self.coordinatePipeline)
        mainThread = threading.Thread(target=self._mainThread_)
        mainThread.start()

    def _mainThread_(self):
        self.coordinates.append(self.coordinatePipeline.getNewPackage())
