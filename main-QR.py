import multiprocessing as mp

from modules.decklinksrc.decklinkSrcWorker_taxi import decklinkSrcWorker_taxi
from modules.QRScanner.QRWorker import qr_worker

if __name__ == "__main__":

    videoPipeline = mp.Queue()

    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target=decklinkSrcWorker_taxi, args=(pause, quit, videoPipeline)),
        mp.Process(target=qr_worker, args=(pause, quit, videoPipeline))
    ]

    for p in processes:
        p.start()
