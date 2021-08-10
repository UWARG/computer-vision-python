import tensorflow as tf
import numpy as np
import multiprocessing as mp
import cv2
import logging
from modules.targetAcquisition.pylonDetection.detect import Detection
from modules.mergeImageWithTelemetry.mergedData import MergedData
from time import sleep

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class TargetAcquisition:
    """
    Performs pylon detection on a given video frame

    ...

    Attributes
    ----------
    boxes : list<yolov2_assets.utils.BoundBox>
        list of bounding boxes that contain tents
    tentCoordinates : dict<yolov2_assets.utils.BoundBox : tuple<xCo, yCo>>
        list of coordinates of the centres of the bounding boxes
    currentFrame : numpy.ndarray
        variable size numpy array (dim: X by Y by 3) of current video frame (given by cv2.imread) (default is a (416, 416, 3) zeros array)
    __logger : Logger
        Program-wide logger


    Methods
    -------
    __init__(deckLinkFrame : np.ndarray, optional)
        Sets given frame to current frame
    set_curr_frame(newFrame : np.ndarray)
        Sets given frame to current frame
    get_coordinates(newFrame : np.ndarray, optional)
        Calls self.__predict() and returns set of coordinates of centres of tents 
    __predict()
        Runs YOLOV2 model on given frame to find tents
    """

    def __init__(self):
        """
        Initializes bbox attributes, sets currentFrame attribute to given frame, zeros otherwise
        """
        self.__logger = logging.getLogger()
        self.__logger.debug("targetAcquisition/__init__: Started")

        # Contains BoundBox objects (see utils.py), each of which contains opposite corners of a rectangle by percentage
        # of height and width of the image as (xmin, ymin) to (xmax, ymax)
        self.bbox=[((0, 0), (0, 0))]
        self.coordinates = []
        self.telemetryData = {}
        self.currentFrame = np.empty(0)
        self.yolo = Detection()

        self.__logger.debug("targetAcquisition/__init__: Finished")

    def set_curr_frame(self, newFrame):
        """
        Sets given frame to current frame

        Parameters
        ----------
        newFrame : MergedData
            MergedData object containing telemetry data & variable size array containing data about a video frame (as given by cv2.imread())
        """
        self.telemetryData = newFrame.telemetry
        self.currentFrame = newFrame.image

    def get_coordinates(self):
        """
        Returns a list of co-ordinates along a video frame where tents are located by running YOLOV5 model
        
        Returns
        -------
        bool
            First returned parameter is a boolean indicating whether the model found bboxes
        tuple
            Returns a two-tuple, where first entry is coordinates, second entry is telemetry data 
        """
        
        # Run YOLOV5 model
        self.__predict()

        # If no bounding boxes found, return False
        if len(self.bbox) == 0:
            return False, []
        
        # Find centre of bounding box
        for i in range(1, len(self.bbox)):
            x = self.bbox[i][0][0] + self.bbox[i][1][0]
            y = self.bbox[i][0][1] + self.bbox[i][1][1]
            self.coordinates.append((x, y))
    
        return True, (self.coordinates, self.telemetryData)
    
    def __predict(self):
        """
        PRIVATE: Runs YOLOV5 model on current frame and populates tentCoordinates and boxes attributes
        """
        # Run YOLOV5 model, put bounding boxes into list
        self.bbox = self.yolo.detect_boxes(self.currentFrame)
        # draw out the bounding boxes
        for (topLeft, botRight) in self.bbox:
            cv2.rectangle(self.currentFrame, topLeft, botRight, (0, 0, 255), 2)
        # cv2.imshow('img', self.currentFrame)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()



