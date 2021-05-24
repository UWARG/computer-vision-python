from QR import QRScanner

def qr_worker(pause, exitRequest, pipelineIn, pipelineOut):
    """
    Initializes worker for QR Scanner module
    """
    print("Start QR Scanner")
    qrScanner = QRScanner()

    while True:
        pause.acquire()
        pause.release()

        frame = pipelineIn.get()
        updatedFrame = qrScanner.main(frame)

        qrScanner.put(updatedFrame)
