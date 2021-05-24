from QR import QRScanner

def qr_worker(pause, exitRequest, pipelineIn, pipelineOut):
    print("Start QR Scanner")
    qrScanner = QRScanner()

    while True:
        pause.acquire()
        pause.release()

        frame = pipelineIn.get()
        updatedFrame = qrScanner.main(frame)

        qrScanner.put(updatedFrame)
