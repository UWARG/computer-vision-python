import cv2 as cv2
import numpy as np

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
        self.__currentFrame = None
        self.capture = cv2.VideoCapture('decklinkvideosrc mode=7 connection=0 ! videoconvert ! appsink') #Starts capture on initialization of object

    def stop(self):
        """
        Logic for stopping video feed by releasing capture and destroying any windows open.
        """
        self.capture.release()
        cv2.destroyAllWindows()
    

    def grab(self, frame):
        """
        Logic for grabbing single frame from DeckLink
        """
        cv2.imshow('Grabbedframe', self.__currentFrame)
    
    def get_frame(self):
        """
        Logic for returning the current frame as a numpy array

        Returns
        -------
        np.ndarray
            Numpy array containing information about the current frame
        """
        return self.__currentFrame

    def recordVideo(self, filename, xdim, ydim):
        """
        Logic for 

        Parameters
        ----------
        filename : str
            Desired name to save file as
        xdim : int
            Width of video frame in pixels
        ydim : int
            Height of video frame in pixels
        """
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
        """
        Logic for displaying video stream from DeckLink
        """
        while(True):
            ret, frame = self.capture.read()
            self.__currentFrame = frame
            cv2.imshow('VideoStream', frame)

            #OpenCV doesn't allow you to access a camera without a camera release, So feel free to replace this bottom with however the video stream will quit (right now it quits on spacebar)
            key = cv2.waitKey(1)
            if key == ord(' '):
                self.quitProgram()
