import cv2 as cv2
import logging
import multiprocessing as mp
import numpy as np
from time import sleep


class DeckLinkSRC:
    """
    Handles DeckLink video stream

    ...

    Attributes
    ----------
    capture : cv2.VideoCapture
        VideoCapture class containing DeckLink stream
    __currentFrame : np.ndarray
        Numpy array storing information about current frame
    __logger : Logger
        Program-wide logger

    Methods
    -------
    __init__()
        Sets up video stream with gstreamer pipeline
    stop()
        Ends video stream
    grab()
        Debug function for showing current frame with cv2.imshow()
    get_frame()
        Returns current frame as numpy array
    display()
        Displays live video feed of DeckLink stream
    """

    def __init__(self):
        """
        Initializes DeckLink video stream
        """
        self.__logger = logging.getLogger()
        self.__logger.debug("decklinksrc/__init__: Started")

        self.__currentFrame = None
        self.capture = None
        self.start()

    def start(self):
        """
        Logic for (re)starting video stream
        """
        self.__currentFrame = None
        self.capture = cv2.VideoCapture(0)  # Starts capture on initialization of object

        self.__logger.debug("decklinksrc/__init__: Finished")

    def stop(self):
        """
        Logic for stopping video feed by releasing capture and destroying any windows open.
        """
        self.__logger.debug("decklinksrc/stop: Started")

        self.capture.release()
        cv2.destroyAllWindows()
        
        self.__logger.debug("decklinksrc/__init__: Finished")

    def grab(self):
        """
        Logic for grabbing single frame from DeckLink
        """
        success, self.__currentFrame = self.capture.read()
        return self.__currentFrame

    def recordVideo(self, filename, xdim, ydim):
        """
        Logic for 
        UNTESTED, use OBS to record

        Parameters
        ----------
        filename : str
            Desired name to save file as
        xdim : int
            Width of video frame in pixels
        ydim : int
            Height of video frame in pixels
        """
        self.__logger.debug("decklinksrc/recordVideo: Started DeckLink Video Record")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 25, (xdim, ydim))

        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        self.__logger.debug("decklinksrc/recordVideo: Stopped DeckLink Video Record")

    def display(self):
        """
        Logic for displaying video stream from DeckLink
        """
        self.__logger.debug("decklinksrc/display: Started DeckLink Video Stream Display")

        while True:
            ret, frame = self.capture.read()
            self.__currentFrame = frame
            cv2.imshow('VideoStream', frame)

            # OpenCV doesn't allow you to access a camera without a camera release, So feel free to replace this
            # bottom with however the video stream will quit (right now it quits on spacebar)
            key = cv2.waitKey(1)
            if key == ord(' '):
                self.stop()
                break
        
        self.__logger.debug("decklinksrc/display: Stopped DeckLink Video Stream Display")
