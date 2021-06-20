from modules.decklinksrc.decklinksrc import DeckLinkSRC


def decklinkSrcWorker(pause, exitRequest, pipelineOut):
    print("Start decklinksrc")
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
            return
          
        pause.acquire()
        pause.release()

        curr_frame = decklinkSrc.grab()
        if curr_frame is None:
            continue

        # Debugging
        # cv2.imshow('VideoStream', curr_frame)
        # cv2.waitKey(1)

        pipelineOut.put(curr_frame)
