from modules.decklinksrc.decklinksrc import DeckLinkSRC
from modules.timestamp.timestamp import Timestamp
import logging

def decklinkSrcWorker(pause, exitRequest, pipelineOut):
    logger = logging.getLogger()
    logger.debug("decklinkSrcWorker: Started DeckLinkSRC module")

    decklinkSrc = DeckLinkSRC()

    # i = 0  # Debugging
    while True:
        # Debugging
        # i += 1
        # if i > 300:
        #     decklinkSrc.stop()
        #     return

        # Kill process if exit is requested
        if not exitRequest.empty():
            decklinkSrc.stop()
            break

        pause.acquire()
        pause.release()

        # Timestamping logic has been implemented in worker to not interfere with Taxi program assumptions
        curr_frame = Timestamp(decklinkSrc.grab())
        if curr_frame is None or curr_frame.data is None:
            continue

        # Debugging
        # cv2.imshow('VideoStream', curr_frame.data)
        # cv2.waitKey(1)

        pipelineOut.put(curr_frame)
    
    logger.debug("decklinkSrcWorker: Stopped DeckLinkSRC module")
