import cv2
import numpy as np
from boxDetection.detect import Detection
from qrScan.scan import scan as scan_qr
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class Taxi:
    """
    Performs cardboard box detection on a given video frame

    Attributes
    ----------
    state : string
        state of object recognition ("BOX", "QR", ...)
    bbox : list<tuple<tuple, tuple>>
        a list of ((x1, y1), (x2, y2)) coordinates for (top left, bottom right) of bounding boxes; one per box
    frame : np.ndarray
        the current video frame
    yolo : Detection object
        the YOLOv5 detector
    tracker : TrackerKCF object
        the cv2 KCF bounding box tracker
    nextUncheckedID : int
        the ID of the next box to scan
    foundBox : bool
        whether the box with the right QR code is found

    Methods
    -------
    __init__()
        Initialize class variables
    set_state(state: string)
        Prepare variables for box detection, qr decoding, etc.
    main()
        Main operations: getting camera input and passing the image to appropriate methods
    """
    
    def __init__(self):
        """
        Initializes variables
        """
        self.state = "BOX"
        self.bbox = [((0, 0), (0, 0))]
        self.frame = []
        self.yolo = Detection()
        self.tracker = cv2.TrackerKCF_create()
        self.nextUncheckedID = 0
        self.foundBox = False

    def set_state(self, state):
        """
        Prepare variables for box detection, qr decoding, etc.

        Parameters
        ----------
        state: string
            state of object recognition ("BOX", "QR", ...)
        """
        if state == "BOX" or state == 0:
            self.state = "BOX"

        elif state == "TRACK" or state == 1:
            self.state = "TRACK"
            # Initialize tracker with first frame and bounding box
            bboxReformat = (self.bbox[0][0][0], self.bbox[0][0][1], self.bbox[0][1][0], self.bbox[0][1][1])
            self.tracker.init(self.frame, bboxReformat)

        elif state == "QR" or state == 2:
            self.state = "QR"

        else:
            print("Error: invalid state selected")

    def main(self):
        """
        Main operations: getting camera input and passing the image to appropriate methods
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, self.frame = cap.read()
            
            if self.state == "BOX":
                self.bbox = self.yolo.detect_boxes(self.frame)
                for (topLeft, botRight) in self.bbox:
                    cv2.rectangle(self.frame, topLeft, botRight, (0,0,255), 2)

            if self.state == "TRACK":
                found, bbox = self.tracker.update(self.frame)
                if found:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    self.bbox = [(p1, p2)]

                    cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)
                else:
                    # TODO: hand off control to human controller when tracking fails
                    print("Tracking failure")

            if self.state == "QR":
                message = scan_qr(self.frame)
                print(message)
                
            cv2.imshow('Image', self.frame)
            
            key = cv2.waitKey(10)
            if key == ord('t') and self.state != "TRACK":
                self.set_state("TRACK")
                print("switch to tracking state")
            if key == ord('q'):
                break

# Instantiate the Taxi object and run operations
if __name__ == '__main__':
    testTaxi = Taxi()
    testTaxi.main()