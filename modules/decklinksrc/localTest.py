import time
import multiprocessing as mp
import decklinkSrcWorker
import cv2

if __name__ = "__main__":

    pause = mp.Lock()
    exitRequest = mp.Queue()
    imagePipeline = mp.Queue()

    # Single-process mode - uncomment the i variable in the worker
    # decklinkSrcWorker.decklinkSrcWorker(pause, exitRequest, imagePipeline)

    # Multiprocess mode
    # """
    p = mp.Process(target=decklinkSrcWorker.decklinkSrcWorker,
                   args=(pause, exitRequest, imagePipeline,))
    p.start()

    time.sleep(3)

    exitRequest.put(True)
    # """
    print("Queue contents:")
    j = 0
    while (True):
        try:
            # A guaranteed delay is necessary, otherwise this loop will collide with itself!
            # I'll investigate how short this can be.
            time.sleep(0.1)
            frame = imagePipeline.get_nowait()
            print(frame)
            j += 1
        except:
            break

    print("Done! " + str(j))
