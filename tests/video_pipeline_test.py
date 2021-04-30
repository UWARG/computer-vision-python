import pytest
import multiprocessing as mp
from modules.targetAcquisition.targetAcquisitionWorker import targetAcquisitionWorker
from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker


# Integration test to check mediator behaviour
# Disable input verification on pipelines to run test
def testVideoPipeline():
    videoPipeline = mp.Queue()
    coordinatePipeline = mp.Queue()
    pause = mp.Lock()
    exitSignal = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker, args=(pause, exitSignal, videoPipeline)),
        mp.Process(target=targetAcquisitionWorker, args=(pause, exitSignal, videoPipeline, coordinatePipeline))
    ]

    for p in processes:
        p.start()

    assert coordinatePipeline.get() is None

    exitSignal.put(True)

