from modules.decklinksrc.decklinksrc import DeckLinkSRC
import logging

def decklinkSrcWorker(pause, exitRequest, pipelineOut):
    logger = logging.getLogger()
    logger.debug("decklinkSrcWorker: Started DeckLinkSRC module")

    decklinkSrc = DeckLinkSRC()

    while True:
        # Kill process if exit is requested
        if not exitRequest.empty():
            break

        pause.acquire()
        pause.release()

        curr_frame = decklinkSrc.grab()
        if curr_frame is None:
            continue

        pipelineOut.put(curr_frame)
    
    logger.debug("decklinkSrcWorker: Stopped DeckLinkSRC module")
