from modules.decklinksrc.decklinksrc import DeckLinkSRC
import logging


def decklinkSrcWorker_taxi(pause, exitRequest, pipelineOut):
    logger = logging.getLogger()
    logger.debug("decklinkSrcWorker: Started DeckLinkSRC module")

    decklinkSrc = DeckLinkSRC()

    while True:

        # Kill process if exit is requested
        if not exitRequest.empty():
            decklinkSrc.stop()
            break

        pause.acquire()
        pause.release()

        # Timestamping logic has been implemented in worker to not interfere with Taxi program assumptions
        curr_frame = decklinkSrc.grab()
        if curr_frame is None:
            continue

        pipelineOut.put(curr_frame)

    logger.debug("decklinkSrcWorker: Stopped DeckLinkSRC module")
