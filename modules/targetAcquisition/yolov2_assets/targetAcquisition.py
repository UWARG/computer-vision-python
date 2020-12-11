import tensorflow as tf
import numpy as np
import cv2
from predict import yolo_predict

class TargetAcquisition:
    #deckLinkFrame is a numpy array of any size containing image pixels
    def __init__ (self, deckLinkFrame=np.zeros((416, 416))):
        self.boxes = [] # Contains BoundBox objects (see utils.py), each of which contains opposite corners of a rectangle by percentage of height and width of the image as (xmin, ymin) to (xmax, ymax)
        self.tentCoordinates = dict()
        self.currentFrame = deckLinkFrame
  
    def predict(self):
        self.boxes = yolo_predict(self.currentFrame)
        self.find_coordinates()
    
    def get_coordinates(self):
        return self.tentCoordinates

    def find_coordinates(self):
        image_h = self.currentFrame.shape[0]
        image_w = self.currentFrame.shape[1]
        for box in self.boxes:
            xmin = int(box.xmin*image_w)
            ymin = int(box.ymin*image_h)
            xmax = int(box.xmax*image_w)
            ymax = int(box.ymax*image_h)

            self.tentCoordinates[box] = ((xmin+xmax)/2, (ymin+ymax)/2)