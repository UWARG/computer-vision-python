import numpy as np

class Geolocation:
    """Locates the geographical position of a set of pixels"""

    # TODO Class members
    def __init__(self):

        self.__cameraOrigin3o = np.array([0.0, 0.0, 2.0])
        self.__cameraDirection3c = np.array([0.0, 0.0, -1.0])
        self.__cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        self.__cameraOrientation3v = 1 * np.cross(self.__cameraDirection3c, self.__cameraOrientation3u)
        self.__referencePixels = [[0, 0],
                                  [0, 1000],
                                  [1000, 0],
                                  [1000, 1000]]

        return

    # TODO Placeholder, add functionality once we figure out how to convert raw plane data
    def convert_input(self):

        return

    # TODO Incomplete
    # Outputs pixel coordinate to geographical coordinate point pairs
    def gather_point_pairs(self):

        # Convert pixels to vectors in world space


        return
