from ..targetAcquisition.targetAcquisition import TargetAcquisition
from ..decklinksrc.decklinksrc import DeckLinkSRC
import multiprocessing as mp
import numpy as np
from time import sleep


class VideoMediator:

    def __init__(self, testMode=False):
        self.videoPipeline = mp.Queue()
        self.coordinatePipeline = mp.Queue()
        self.coordinates = mp.Manager().list()
        DeckLinkSRC(self.videoPipeline)
        TargetAcquisition(self.videoPipeline, self.coordinatePipeline)
        mainProcess = mp.Process(target=self._mainProcess_, daemon=True, args=(testMode, ))
        mainProcess.start()
        mainProcess.join()

    def get_coordinates(self):
        return list(self.coordinates)

    def _mainProcess_(self, testMode):
        print("main process started")
        while True:
            pipelineLatest = self.coordinatePipeline.get()
            if pipelineLatest not in self.coordinates:
                self.coordinates.append(pipelineLatest)

                if testMode:
                    break

            sleep(0.1)

