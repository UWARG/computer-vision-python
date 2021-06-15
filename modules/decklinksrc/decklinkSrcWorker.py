from modules.decklinksrc.decklinksrc import DeckLinkSRC


def decklinkSrcWorker(pause, exitRequest, pipelineOut):
    print("Start decklinksrc")
    decklinkSrc = DeckLinkSRC()

    while True:
        # Kill process if exit is requested
        if not exitRequest.empty():
            return

        pause.acquire()
        pause.release()

        curr_frame = decklinkSrc.grab()
        if curr_frame is None:
            continue

        pipelineOut.put(curr_frame)


