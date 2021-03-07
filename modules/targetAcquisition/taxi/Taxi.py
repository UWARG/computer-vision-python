from boxDetection.detect import detect
from qrDetection.scan import decoder as qr_scan

class Taxi:
    """
    Performs cardboard box detection on a given video frame

    Attributes
    ----------
    boundingBoxes : list<tuple<tuple, tuple>>
        List of bounding boxes 

    Methods
    -------
    __init__()
        Initialize class variables
    predict_boxes()
        Get bounding box of cardboard boxes from YOLOv5
    read_qr(image : np.ndarray)
        Read message from QR code
    """
    
    def __init__(self):
        """
        Initializes variables
        """
        self.boundingBoxes = []

    def predict_boxes(self):
        """
        Get bounding box of cardboard boxes from YOLOv5
        """
        # TODO: Get detect.py to return coordinates of each cardboard box after predicting from one video frame
        # TODO: Decide on a return format:
            # Formats: [((x0, y0), (x1, y1))] or [((x, y), (w, h))]
        # TODO: reduce time between detect launch and image detection
        self.boundingBoxes = detect()

    def read_qr(self, image):
        """
        Read QR code from image

        Parameters
        ----------
        image : np.ndarray
            Image containing a single readable QR code
        """
        # TODO: Get qrDetection scan.py to return the QR message as string instead of displaying it
        self.qrMessage = qr_scan(image)


# TODO: Set up video feed to feed image into __predict_boxes() and __read_qr()
'''
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    decoder(frame)
    cv2.imshow('this shouldn't be showing up', frame)
    code = cv2.waitKey(10)
    if code == ord('q'):
        break
'''

if __name__ == '__main__':
    testTaxi = Taxi()
    testTaxi.predict_boxes()