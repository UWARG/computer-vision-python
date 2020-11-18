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
    
    def grab(self): #Logic for grabbing frame from deckLink


    