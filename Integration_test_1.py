import multiprocessing as mp
import pytest
import cv2
from modules.TestIMUINterface.getIMUData import getIMUINterface
from modules.decklinksrc.decklinksrc import DeckLinkSRC

def test_decklink_TestIMUInterface():

    decklinkSrc = DeckLinkSRC()
    getIMUInterface = getIMUINterface()

    while True:
        curr_frame = decklinkSrc.grab()
        #assert curr_frame != None

        #decklinkSrc.display()

        temp = getIMUINterface.getIMUData("COM3")

        if temp != None:
            break

    print(temp)

test_decklink_TestIMUInterface()









    # videoPipeline = mp.Queue()
    # # Queue from command module out to fusion module containing timestamped telemetry data from POGI

    # # Utility locks
    # pause = mp.Lock()
    # quit = mp.Queue()

    # processes = [
    #     mp.Process(target=decklinkSrcWorker_taxi, args=(pause, quit, videoPipeline)),
    #     mp.Process(target=getIMUINterface.getIMUData, args=(videoPipeline))
    # ]

    # for p in processes:
    #     p.start()

    # #logger.debug("main/showVideo: Video Display Finished")
    # return


