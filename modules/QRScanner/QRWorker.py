from modules.QRScanner.QR import QRScanner
import logging
import cv2

def qr_worker(pause, exitRequest, pipelineIn):
    """
    Initializes worker for QR Scanner module
    """
    logger = logging.getLogger()
    logger.debug("QRWorker: Start QR Scanner Module")
    
    qrScanner = QRScanner()

    while True:
        pause.acquire()
        pause.release()

        frame = pipelineIn.get()
        updatedFrame, output = qrScanner.main(frame)

        if (output is not None):
            print(output)

        cv2.imshow("Video feed", updatedFrame)

        cv2.waitKey(1)
        
        if not exitRequest.empty():
            break

    logger.debug("QRWorker: Stop QR Scanner Module")
