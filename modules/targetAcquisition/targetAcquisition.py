import tensorflow as tf
import numpy as np
import multiprocessing as mp
from .yolov2_assets.predict import yolo_predict
from time import sleep


class TargetAcquisition:
    """
    Performs object detection on a given video frame

    ...

    Attributes
    ----------
    boxes : list<yolov2_assets.utils.BoundBox>
        list of bounding boxes that contain tents
    tentCoordinates : dict<yolov2_assets.utils.BoundBox : tuple<xCo, yCo>>
        list of coordinates of the centres of the bounding boxes
    currentFrame : numpy.ndarray
        variable size numpy array (dim: X by Y by 3) of current video frame (given by cv2.imread) (default is a (416, 416, 3) zeros array)


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
    __find_coordinates()
        Interprets BoundBox class values to populate self.tentCoordinates with centres of tents
    """

    def __init__(self):
        """
        Initializes boxes, tentCoordinates attributes, sets currentFrame attribute to given frame, zeros otherwise
        """

        # Contains BoundBox objects (see utils.py), each of which contains opposite corners of a rectangle by percentage
        # of height and width of the image as (xmin, ymin) to (xmax, ymax)
        self.boxes = []
        self.tentCoordinates = dict()
        self.currentFrame = np.empty(0)

    def set_curr_frame(self, newFrame):
        """
        Sets given frame to current frame

        Parameters
        ----------
        newFrame : np.ndarray
            Variable size array containing data about a video frame (as given by cv2.imread())
        """
        self.currentFrame = newFrame

    def get_coordinates(self, newFrame=np.zeros((416, 416, 3))):
        """
        Returns a list of co-ordinates along a video frame where tents are located by running YOLOV2 model

        Parameters
        ----------
        newFrame : np.ndarray, optional
            Variable size array containing data about a video frame (as given by cv2.imread())
        
        Returns
        -------
        dict<yolov2_assets.utils.BoundBox : tuple<xCo, yCo>>
            Returns a dict of bounding boxes where tents are located, with centre coordinates for each
        """
        # If new frame has been specified (is non-empty, set the current frame to the given frame)
        if np.count_nonzero(newFrame) != 0:
            self.set_curr_frame(newFrame)
        else:
            return
        # Run YOLOV2 model
        self.__predict()
        return self.tentCoordinates

    def __predict(self):
        """
        PRIVATE: Runs YOLOV2 model on current frame and populates tentCoordinates and boxes attributes
        """
        # Run YOLOV2 model, put bounding boxes into list
        self.boxes = yolo_predict(self.currentFrame)
        # Find centre coordinates for each bounding box
        self.__find_coordinates()

    def __find_coordinates(self):
        """
        PRIVATE: Finds centre coordinates for each bounding box, populates tentCoordinates
        """
        image_h = self.currentFrame.shape[0]
        image_w = self.currentFrame.shape[1]
        for box in self.boxes:
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)

            self.tentCoordinates[box] = ((xmin + xmax) / 2, (ymin + ymax) / 2)

# Testing
# tracker = TargetAcquisition(cv2.imread('yolov2_assets/single_test_images/raccoon-1.jpg'))
# print(tracker.get_coordinates())
