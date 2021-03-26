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
    expectedCount : int
        the number of boxes it needs to detect to go into tracking state
    expectedQR : string
        QR code on the correct box
    numStableFrames : int
        number of consecutive frames with all <expectedCount> number of boxes in view before switching to TRACK

    Methods
    -------
    __init__()
        Initialize class variables
    set_state(state: string)
        Prepare variables for box detection, qr decoding, etc.
    main()
        Main operations: getting camera input and passing the image to appropriate methods
    """
    
    def __init__(self, state = "BOX", bbox = [((0, 0), (0, 0))], frame = [], 
                 nextUncheckedID = 0, expectedCount = 5, expectedQR = "abcde12345", stableFrames = 20):
        """
        Initializes variables
        """
        self.state = state
        self.bbox = bbox
        self.frame = frame
        self.yolo = Detection()
        self.tracker = None
        self.nextUncheckedID = nextUncheckedID
        self.expectedCount = expectedCount
        self.expectedQR = expectedQR
        self.numStableFrames = stableFrames

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
            bbox = self.bbox[self.nextUncheckedID]
            bboxReformat = (bbox[0][0], bbox[0][1], 
                            bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
            
            # Note: must create the tracker here instead of __init__, else can't switch freely between track & box 
            self.tracker = cv2.TrackerKCF_create()
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
        
        # The number of consecutive frames where all <expectedCount> boxes are in view
        frameCount = 0
        
        # Wait for a certain number of detections before issuing the next movement command
        moveWaitCount = 0
        while True:
            ret, self.frame = cap.read()
            
            if self.state == "BOX":
                self.bbox = self.yolo.detect_boxes(self.frame)
                for (topLeft, botRight) in self.bbox:
                    cv2.rectangle(self.frame, topLeft, botRight, (0,0,255), 2)
                
                # If lose track of 1+ boxes, reset stable frame count
                if len(self.bbox) == self.expectedCount:
                    frameCount += 1
                else:
                    frameCount = 0
                
                if frameCount == self.numStableFrames:
                    self.set_state("TRACK")
                    frameCount = 0

            if self.state == "TRACK":
                found, bbox = self.tracker.update(self.frame)
                if found:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    self.bbox = [(p1, p2)]
                    cv2.rectangle(self.frame, p1, p2, (255, 0, 0), 2, 1)

                    if moveWaitCount < 10:
                        moveWaitCount += 1
                    else:
                        # Don't need to turn if the bbox midpoint is within the center margin
                        # If center to left past margin, turn right by 1 degree
                        # If center to right past margin, turn left by 1 degree
                        margin = 50
                        turn = 0
                        forward = 0
                        if (p1[0] + p2[0])/2 < self.frame.shape[0]/2 - margin:
                            turn = 1
                        elif (p1[0] + p2[0])/2 > self.frame.shape[0]/2 + margin:
                            turn = -1
                        else:
                            forward = 5
                        print(f"Turn {turn} degree(s) and move forward by {forward}")
                        moveWaitCount = 0

                        # TODO: fix self.frame.shape not matching cv.imshow video shape
                        cv2.rectangle(self.frame, (240,0), (240,640), (255, 0, 255), 2, 1)
                
                else:
                    # TODO: hand off control to human controller when tracking fails
                    # Without forgetting what was the last unchecked ID
                    # Maybe set that as an __init__ parameter? 
                    print("Tracking failure. Switch to human control")

            if self.state == "QR":
                message = scan_qr(self.frame)
                # print(message)
                if message != None:
                    if message == self.expectedQR:
                        print("QR matches. Pick up the box")
                    else:
                        print("QR does not match. Turn around and try again")
                
            cv2.imshow('Image', self.frame)
            
            # Temporary manual box selection
            key = cv2.waitKey(10)
            if key == ord('b') and self.state != "BOX":
                self.set_state("BOX")
                print("switch to box state")

            # Temporary manual tracking selection
            if key == ord('t') and self.state != "TRACK":
                self.set_state("TRACK")
                print("switch to tracking state")
            
            # Temporary manual QR code selection
            if key == ord('r') and self.state != "QR":
                self.set_state("QR")
                print("switch to QR state")
            
            if key == ord('q'):
                break

# Instantiate the Taxi object and run operations
if __name__ == '__main__':
    testTaxi = Taxi(expectedQR = "abcde12345")
    testTaxi.main()