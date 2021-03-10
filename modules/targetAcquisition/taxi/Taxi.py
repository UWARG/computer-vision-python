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
        self.state = None

    def set_state(self, state):
        """
        Prepare variables for box detection, qr decoding, etc.

        Parameters
        ----------
        state: string
            state of object recognition ("BOX", "QR", ...)
        """
        if state == "BOX":
            # TODO: set other variables here if necessary
            self.state = "BOX"

        elif state == "QR":
            # TODO: set other variables here if necessary
            self.state = "QR"

        else:
            print("Error: invalid state selected")

    def main(self):
        """
        Main operations: getting camera input and passing the image to appropriate methods
        """
        cap = cv2.VideoCapture(0)
        yolo = Detection()
        
        while True:
            ret, frame = cap.read()
            
            # TODO: wrap this step in a preprocessing function
            if self.state == "BOX":
                boundingBoxes = yolo.detect_boxes(img = frame)
                
                for (topLeft, botRight) in boundingBoxes:
                    frame = cv2.rectangle(frame, topLeft, botRight, (0,0,255), 2)

            # TODO: wrap this step in a preprocessing function
            if self.state == "QR":
                message = scan_qr(img = frame)
                print(message)
                
            cv2.imshow('Image', frame)
            
            if cv2.waitKey(10) == ord('q'):
                break

# Instantiate the Taxi object and run operations
if __name__ == '__main__':
    testTaxi = Taxi()
    testTaxi.set_state(state = "BOX")
    testTaxi.main()