from ..targetAcquisition.targetAcquisition import TargetAcquisition
from ..decklinksrc.decklinksrc import DeckLinkSRC
from ...pipelines.coordinatePipeline import CoordinatePipeline
from ...pipelines.videoPipeline import VideoPipeline
import threading


class VideoMediator:

    def __init__(self):
        self.videoPipeline = VideoPipeline()
        self.coordinatePipeline = CoordinatePipeline()
        self.coordinates = []
        DeckLinkSRC(self.videoPipeline)
        TargetAcquisition(self.videoPipeline, self.coordinatePipeline)
        main = threading.Thread(target=self._main_)
        main.start()

    def _main_(self):
        self.coordinates.append(self.coordinatePipeline.getNewPackage())
