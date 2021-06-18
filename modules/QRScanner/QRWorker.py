from QR import QRScanner
import cv2

def qr_worker(pause, exitRequest, pipelineIn):
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

        cv2.imshow("Video feed", updatedFrame)
        
        if not exitRequest.empty():
            return
