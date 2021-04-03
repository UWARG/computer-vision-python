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

    def calculate_pixel_to_geo_mapping(self):
        """
        Outputs transform matrix for mapping pixels to geographical points

        Returns
        -------
        np.array(shape=(3,3))
        """

        # Declare 4 matrices
        # Assign relevant values, shapes and data types
        # Create a 3x3 matrix with the coordinates as vectors with 1 as the z component => np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
        sourcePixelMatrix = np.vstack((self.__pixelToGeoPairs[0:3, 0:1].reshape(3, 2).T, [1, 1, 1])).astype(np.float64)
        sourcePixelVector = np.vstack((self.__pixelToGeoPairs[3, 0:1].reshape(1, 2).T, [1])).astype(np.float64)
        mappedGeoMatrix = np.vstack((self.__pixelToGeoPairs[0:3, 1:2].reshape(3, 2).T, [1, 1, 1])).astype(np.float64)
        mappedGeoVector = np.vstack((self.__pixelToGeoPairs[3, 1:2].reshape(1, 2).T, [1])).astype(np.float64)

        # Solve system of linear equations to get value of coefficients
        solvedPixelVector = np.linalg.solve(sourcePixelMatrix, sourcePixelVector)
        solvedGeoVector = np.linalg.solve(mappedGeoMatrix, mappedGeoVector)

        # Multiply coefficients with corresponding columns in matrices sourcePixelMatrix and mappedGeoMatrix
        for i in range(0, 3):
            sourcePixelMatrix[:, i] *= solvedPixelVector[i][0]
            mappedGeoMatrix[:, i] *= solvedGeoVector[i][0]

        # Invert sourcePixelMatrix
        # Using pinv() instead of inv() for handling ill-conditioned matrices
        sourcePixelMatrixInverse = np.linalg.pinv(sourcePixelMatrix)

        # Return matrix product of mappedGeoMatrix and sourcePixelMatrixInverse
        return (mappedGeoMatrix.dot(sourcePixelMatrixInverse))

# Private function that checks if the trimmed array is empty ie all values were too far apart from the median
# Returns median if array is empty, otherwise finds the average of the trimmed array
    def __get_average_otherwise_median(self,trimmedArray,median):
        if np.size(trimmedArray,0)==0:
             return median
        else:
            return np.average(trimmedArray,axis=0)


    def get_best_location(self,inputLocationTupleList):

        # For the  case of a single row matrix being passed to the function
        if np.size(inputLocationTupleList,0)==1:

            averagePair = np.vstack(inputLocationTupleList[:, 0]).astype(np.float64)
            averageError = np.vstack(inputLocationTupleList[:, 1]).astype(np.float64)


        else:

            # Splits the 3D numpy arrray into three separate arrays
            coordPair = np.vstack(inputLocationTupleList[:, 0]).astype(np.float64)
            errorArray = np.vstack(inputLocationTupleList[:, 1]).astype(np.float64)
            confidenceArray = np.vstack(inputLocationTupleList[:, 2]).astype(np.float64)

            # Splits coordinate pair into x-coord and y-coord to remove outliers
            xCoord = np.vstack(coordPair[:, [0][0]]).astype(np.float64)
            yCoord = np.vstack(coordPair[:, [1][0]]).astype(np.float64)
            xCoordTrimmed=[]
            yCoordTrimmed=[]

            xCoordMedian = np.median(xCoord)
            yCoordMedian = np.median(yCoord)
            errorMedian = np.median(errorArray)

            # A modified version of the trimmed mean calculation. Here the distance measurement is distance between every x,y point from the median coordinate pair, x median and y median.
            # 7 is the distance for how far a given point can be from the median
            # TODO: Z-scores could be used instead
            for (x,y) in zip(xCoord,yCoord):
                distance = math.hypot(xCoordMedian-x,yCoordMedian-y)

                if distance < 7:
                    xCoordTrimmed.append(x)
                    yCoordTrimmed.append(y)

            # In the case of the values in the error array, the distance here is 3.5.
            errorArray = errorArray[(errorArray - errorMedian > -3.5) & (errorArray - errorMedian < 3.5)]

            averageX = self.__get_average_otherwise_median(xCoordTrimmed,xCoordMedian)
            averageY = self.__get_average_otherwise_median(yCoordTrimmed,yCoordMedian)
            averageError =self.__get_average_otherwise_median(errorArray,errorMedian)

            averagePair = (averageX,averageY)

        return (averagePair,averageError)
    
