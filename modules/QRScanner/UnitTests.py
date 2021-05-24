import imutils
from imutils.video import VideoStream
import QR
import pyzbar from pyzbar
import cv2

class QRTest:
    def __init__(self):
        
        self.scanner = QR.QRScanner()

        imagePath = "qrtest.jpg"
        self.run_single_image_test(imagePath)
        
        self.videoStream = VideoStream(src=0).start()
        self.run_video_test()
    
    def run_single_image_test(self, imagePath):
        image = cv2.imread(imagePath)
        cv2.imshow(self.scanner.main(image))

        while True:
            ans = input("Type 'q' to exit")
            if ans == 'q':
                break

    def run_video_test(self):
        print("Press q to exit the stream")
        while True:
            frame = self.videoStream.read()
            frame = imutils.resize(frame, width=400)

            self.scanner.main(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

if __name__ == "__main__":
    QRTest()