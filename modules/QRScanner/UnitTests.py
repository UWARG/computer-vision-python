import imutils
from imutils.video import VideoStream
import QR
import time
from pyzbar import pyzbar
import cv2

class QRTest:
    def __init__(self):
        
        self.scanner = QR.QRScanner()

        imagePath = "qrtest.png"
        self.run_single_image_test(imagePath)
        
        self.videoStream = VideoStream(src=0).start()
        self.run_video_test()
    
    def run_single_image_test(self, imagePath):
        image = cv2.imread(imagePath)
        image = self.scanner.main(image)
        cv2.imshow('single_image', image)

        print("Press q to close")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    def run_video_test(self):
        print("Press q to exit the stream")
        while True:
            frame = self.videoStream.read()
            frame = imutils.resize(frame)

            frame = self.scanner.main(frame)
            cv2.imshow('video', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
        
        self.videoStream.stop()

if __name__ == "__main__":
    QRTest()
