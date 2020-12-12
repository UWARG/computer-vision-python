import cv2 as cv2
import numpy as np

class DeckLinkSRC:

    def __init__(self):
        self.__currentFrame = None
        self.capture = cv2.VideoCapture('decklinkvideosrc mode=7 connection=0 ! videoconvert ! appsink') #Starts capture on initialization of object
        #Because of this, we no longer need start stream code from Aryan

    def stop(self): #Logic for stopping video feed by releasing capture and destroying any windows open.
        self.capture.release()
        cv2.destroyAllWindows()
    
    def grab(self): #Logic for displaying single frame from deckLink
        cv2.imshow('Grabbedframe', self.__currentFrame)
    
    def get_frame(self): #Logic for returning the current frame as a numpy array
        return self.__currentFrame
    
    def quitProgram(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def display(self):
        while(True):
            ret, frame = self.capture.read()
            self.__currentFrame = frame
            cv2.imshow('VideoStream', frame)

            #OpenCV doesn't allow you to access a camera without a camera release, So feel free to replace this bottom with however the video stream will quit (right now it quits on spacebar)
            key = cv2.waitKey(1)
            if key == ord(' '):
                self.quitProgram()