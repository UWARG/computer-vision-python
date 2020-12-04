import cv2 as cv2
import numpy as np

class DeckLinkSRC:

    def __init__(self):
        self.currentFrame
        self.capture = cv2.VideoCapture('decklinkvideosrc mode=7 connection=0 ! videoconvert ! appsink') #Starts capture on initialization of object
        #Because of this, we no longer need start stream code from Aryan

    def stop(self): #Logic for stopping video feed by releasing capture and destroying any windows open.
        self.capture.release()
        cv2.destroyAllWindows()
    
    def grab(self, frame): #Logic for grabbing frame from deckLink
        self.cv2.imshow('Grabbedframe', frame)
    
    def quitProgram(self):
        self.capture.release()
        self.cv2.destroyAllWindows()


    # We need the file name, x dimensions, and y dimensions
    def recordVideo(self, filename, xdim, ydim):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 25, (xdim, ydim) )

        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            if ret == True:
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


    def display(self):
        while(True):
            ret, frame = self.capture.read()
            cv2.imshow('VideoStream', frame)

            #OpenCV doesn't allow you to access a camera without a camera release, So feel free to replace this bottom with however the video stream will quit (right now it quits on spacebar)
            key = cv2.waitKey(1)
            if key == ord(' '):
                self.quitProgram()
