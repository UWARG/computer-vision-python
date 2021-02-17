"""
Geolocation module to map pixel coordinates to geographical coordinates
"""

import numpy as np

class Geolocation:
    """
    Locates the geographical position of a set of pixels
    """

    # TODO Class members
    def __init__(self):
        """
        Initial setup of class members

        Returns
        -------
        Geolocation
        """

        # Input to gather_point_pairs()
        self.__cameraOrigin3o = np.array([0.0, 0.0, 2.0])
        self.__cameraDirection3c = np.array([0.0, 0.0, -1.0])
        self.__cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        self.__cameraOrientation3v = 1 * np.cross(self.__cameraDirection3c, self.__cameraOrientation3u)
        self.__cameraResolution = np.array([1000, 1000])  # TODO Make global?
        self.__referencePixels = np.array([[0, 0],
                                           [0, 1000],
                                           [1000, 0],
                                           [1000, 1000]])

        # Output of gather_point_pairs()
        # Input to calculate_pixel_to_geo_mapping()
        self.__pixelToGeoPairs = np.array([[[0, 0], [0, 0]],
                                           [[1, 1], [1, 1]],
                                           [[2, 2], [2, 2]],
                                           [[3, 3], [3, 3]]])

        return


    # TODO Placeholder, add functionality once we figure out how to convert raw plane data
    def convert_input(self):
        """
        Converts plane data into data usable by Geolocation

        Returns
        -------

        """

        return


    def gather_point_pairs(self):
        """
        Outputs pixel-geographical coordinate point pairs from camera position and orientation

        Returns
        -------
        # np.array(shape=(n, 2, 2))
        """

        pixelGeoPairs = np.empty(shape=(0, 2, 2))
        minimumPixelCount = 4  # Required for creating the map
        validPixelCount = self.__referencePixels.shape[0]  # Current number of valid pixels (number of rows)
        maximumZcomponent = -0.1  # This must be lesser than zero and determines if the pixel is pointing downwards

        # Find corresponding geographical coordinate for every valid pixel
        for i in range(0, self.__referencePixels.shape[0]):

            # Not enough pixels to create the map, abort
            if (validPixelCount < minimumPixelCount):
                return np.empty(shape=(0, 2, 2))

            # Convert current pixel to vector in world space
            pixel = self.__referencePixels[i]
            # Scaling in the u, v direction
            scalar1m = 2 * pixel[0] / self.__cameraResolution[0] - 1
            scalar1n = 2 * pixel[1] / self.__cameraResolution[1] - 1

            # Linear combination formula
            pixelInWorldSpace3a = self.__cameraDirection3c + scalar1m * self.__cameraOrientation3u + scalar1n * self.__cameraOrientation3v

            # Verify pixel vector is pointing downwards
            if (pixelInWorldSpace3a[2] > maximumZcomponent):
                validPixelCount -= 1
                continue

            # Find intersection of the pixel line with the xy-plane
            x = self.__cameraOrigin3o[0] - pixelInWorldSpace3a[0] * self.__cameraOrigin3o[2] / pixelInWorldSpace3a[2]
            y = self.__cameraOrigin3o[1] - pixelInWorldSpace3a[1] * self.__cameraOrigin3o[2] / pixelInWorldSpace3a[2]

            # Insert result
            pair = np.vstack((self.__referencePixels[i], [x, y]))
            pixelGeoPairs = np.concatenate((pixelGeoPairs, [pair]))

        return pixelGeoPairs
