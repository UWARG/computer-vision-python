import cv2
import imutils
from imutils.video import VideoStream
import time

from pyzbar import pyzbar

import QR

class QRTest:
    """
    Unit Tests for QR.py

    ...

    Attributes
    ----------
    scanner : QRScanner
        QR.py module
    videoStream : imutils.video.VideoStream
        Incoming video to be read
    
    Methods
    -------
    __init__()
        Initializes module, runs single image test & video test
    run_single_image_test(imagePath : str)
        Runs QR Scanner on a single, static image
    run_video_test()
        Runs QR Scanner on live video feed
    """
    def __init__(self):
        """
        Initializes module, runs single image test & video test
        """
        self.scanner = QR.QRScanner()

        # imagePath = "qrtestnew.png"
        # self.run_single_image_test(imagePath)
        
        self.videoStream = VideoStream(src=0).start()
        self.run_video_test()
    
    def run_single_image_test(self, imagePath):
        """
        Runs QR Scanner on a single, static image

        Parameters
        ----------
        imagePath : str
            Relative path to the image to be tested
        """
        image = cv2.imread(imagePath)
        image, output = self.scanner.main(image)

        print(output)

        #print(self.scanner.get_qr_text()) #HERE

        cv2.imshow('single_image', image)

        print("Press q to close")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    def run_video_test(self):
        """
        Runs QR Scanner on live video feed
        """
        test = None

        print("Press q to exit the stream")
        while True:
            frame = self.videoStream.read()
            frame = imutils.resize(frame)

            frame, output = self.scanner.main(frame)
            cv2.imshow('video', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

            if output != None and test == None:
                test = output

        print (test)
        
        self.videoStream.stop()

if __name__ == "__main__":
    QRTest()
