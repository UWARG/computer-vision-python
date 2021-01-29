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
        self.__cameraResolution = [1000, 1000]  # TODO Make global?

        return

    # TODO Placeholder, add functionality once we figure out how to convert raw plane data
    def convert_input(self):

        return

    # TODO Incomplete
    # Outputs pixel coordinate to geographical coordinate point pairs
    def gather_point_pairs(self):

        # Convert pixels to vectors in world space

        pixels = np.array(self.__referencePixels).T

        # Get u, v scaling from pixel coordinate and resolution
        m = 2 * pixels[0] / self.__cameraResolution[0] - 1
        m = np.atleast_2d(m).T
        n = 2 * pixels[1] / self.__cameraResolution[1] - 1
        n = np.atleast_2d(n).T

        # Duplicate the vectors for numpy
        points = len(self.__referencePixels)
        c = np.tile(self.__cameraDirection3c, (points, 1))
        u = np.tile(self.__cameraOrientation3u, (points, 1))
        v = np.tile(self.__cameraOrientation3v, (points, 1))

        # Formula
        pixelsInWorldSpace = c + m * u + n * v


        # Find intersection of the pixel line with the xy-plane



        return
