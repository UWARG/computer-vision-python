from modules.decklinksrc.decklinksrc import DeckLinkSRC


def decklinkSrcWorker(pause, exitRequest, pipelineOut):
    print("Start decklinksrc")
    decklinkSrc = DeckLinkSRC()
    while True:
        pause.acquire()
        pause.release()

        curr_frame = decklinkSrc.grab()
        pipelineOut.put(curr_frame)


        if not exitRequest.empty():
            return
