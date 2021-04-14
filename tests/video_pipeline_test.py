import pytest
import multiprocessing as mp
from modules.targetAcquisition.targetAcquisitionWorker import targetAcquisitionWorker
from modules.decklinksrc.decklinkSrcWorker import decklinkSrcWorker


# Integration test to check mediator behaviour
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


def testPipelinePause():
    pipeline = mp.Queue()
    pause = mp.Lock()
    exitSignal = mp.Queue()

    p = mp.Process(target=decklinkSrcWorker, args=(pause, exitSignal, pipeline))
    p.start()

    last_len = pipeline.qsize()
    pause.acquire()
    new_len = pipeline.qsize()
    pause.release()

    assert last_len == new_len

    exitSignal.put(True)
