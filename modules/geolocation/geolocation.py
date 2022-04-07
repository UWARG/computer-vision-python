"""
Geolocation module to map pixel coordinates to geographical coordinates
"""

import numpy as np
import math
import logging

class Geolocation:
    """
    Locates the geographical position of a set of pixels
    """

    # TODO Class members
    def __init__(self, gpsCoordinates={"latitude": 0, "longitude": 0, "altitude": 0}, eulerCamera={}, eulerPlane={}):
        """
        Initial setup of class members

        Parameters
        ----------
        gpsCoordinates: dict
            Dictionary that contains GPS coordinate data with key names latitude, longitude, and altitude
        eulerCamera: dict
            Dictionary that contains a set of three euler angles for the camera with key names z, y, x
        eulerPlane: dict
            Dictionary that contains a set of three euler angles for the plane with key names z, y, x

        Returns
        -------
        Geolocation
        """

        self.__logger = logging.getLogger()
        self.__logger.debug("geolocation/__init__: Started")

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

        # Inputs to input_conversion
        self.__latitude = gpsCoordinates["latitude"]
        self.__longitude = gpsCoordinates["longitude"]
        self.__altitude = gpsCoordinates["altitude"]
        self.__eulerCamera = eulerCamera
        self.__eulerPlane = eulerPlane

        # Constants for input_conversion
        self.__WORLD_ORIGIN = np.empty(3)
        self.__GPS_OFFSET = np.empty(3)
        self.__CAMERA_OFFSET = np.empty(3)
        self.__FOV_FACTOR_H = 1
        self.__FOV_FACTOR_V = 1
        self.__C_VECTOR_CAMERA_SPACE = [1, 0, 0]
        self.__U_VECTOR_CAMERA_SPACE = [0, 1, 0]

        self.__locationsList = []

        self.__logger.debug("geolocation/__init__: Finished")
        return


    def set_constants(self):
        """
        Magic numbers for competition
        """
        self.__GPS_OFFSET = np.zeros(3)
        self.__CAMERA_OFFSET = np.zeros(3)
        self.__WORLD_ORIGIN = np.zeros(3)
        self.__FOV_FACTOR_H = np.tan(np.deg2rad([85.8 / 2]))
        self.__FOV_FACTOR_V = np.tan(np.deg2rad([55.2 / 2]))

        self.__LAT_ORIGIN = 43.43592232053646
        self.__LON_ORIGIN = -80.58007312309068
        self.__EARTH_RADIUS = 6368073  # From https://planetcalc.com/7721/


    # Requires set_constants() first
    # TODO Unit tests
    def local_from_lat_lon(self, latitude, longitude):
        """
        Get metres from longitude and latitude
        x-axis is east, y-axis is north
        Equations from https://www.themathdoctors.org/distances-on-earth-3-planar-approximation/

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        y = np.deg2rad([latitude - self.__LAT_ORIGIN])[0] * self.__EARTH_RADIUS
        x = np.deg2rad([longitude - self.__LON_ORIGIN])[0] * self.__EARTH_RADIUS * np.cos(np.deg2rad([self.__LAT_ORIGIN]))[0]

        return x, y


    # TODO Unit tests and description
    def lat_lon_from_local(self, x, y):

        latitude = np.rad2deg([y / self.__EARTH_RADIUS])[0] + self.__LAT_ORIGIN
        longitude = np.rad2deg([x / self.__EARTH_RADIUS / np.cos(np.deg2rad([self.__LAT_ORIGIN]))[0]])[0] + self.__LON_ORIGIN

        return latitude, longitude


    def gather_point_pairs(self):
        """
        Outputs pixel-geographical coordinate point pairs from camera position and orientation

        Returns
        -------
        # np.array(shape=(n, 2, 2))
        """
        self.__logger.debug("geolocation/gather_point_pairs: Started")

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

        self.__logger.debug("geolocation/gather_point_pairs: Returned " + str(pixelGeoPairs))
        return pixelGeoPairs

    def __are_three_points_collinear(self, p1, p2, p3):
        """
        PRIVATE
        Evaluates whether the three given points are collinear

        Parameters
        ----------
        p1 : tuple/np.ndarray/list
            Point 1, in form (x, y)
        p2 : tuple/np.ndarray/list
            Point 2, in form (x, y)
        p3 : tuple/np.ndarray/list
            Point 3, in form (x, y)

        Returns
        -------
        bool
            True if the three points are collinear, otherwise false
        """
        self.__logger.debug("geolocation/__are_three_points_collinear: Started")
        
        x = 0
        y = 1
        # Calculates the area of a triangle and checks if this value is 0
        # Actually calculates 2 times the area, since it skips the unnecessary step of multiplication by 0.5
        isCollinear =  (p1[x]*(p2[y]-p3[y]) +
                        p2[x]*(p3[y]-p1[y]) +
                        p3[x]*(p1[y]-p2[y])) == 0

        self.__logger.debug("geolocation/__are_three_points_collinear: Returned " + str(isCollinear))
        return isCollinear

    def get_non_collinear_points(self, coordinatesArray):
        """
        Returns a list of four coordinates from an input array that are not collinear to one another.

        Parameters
        ----------
        coordinatesArray : np.ndarray
            Array with dimensions ( , 2), containing a list of coordinates

        Returns
        -------
        np.ndarray
            Array with dimensions (4, 2), containing a list of coordinates that are non-collinear,
            or an empty list if none were found in the input array
        """
        self.__logger.debug("geolocation/get_non_collinear_points: Started")

        NUM_POINTS_NEEDED = 4
        indexes = np.empty(shape=(NUM_POINTS_NEEDED), dtype = int)

        # If there aren't four points, return the empty array
        if len(coordinatesArray) < NUM_POINTS_NEEDED:
            self.__logger.debug("geolocation/get_non_collinear_points: Returned np.empty(shape=(0,2))")
            return np.empty(shape=(0, 2))

        # Look at all sequential pairs of four points
        for i in range(0, len(coordinatesArray)):
            # Array for storing the four points currently being considered
            points = np.empty(shape=(NUM_POINTS_NEEDED, 2))

            # Append four sequential points to the array, loop around to index 0 if needed
            # For efficiency, this algorithm will check through sequential sets of points only, rather than
            # testing every single combination.
            for j in range(0, NUM_POINTS_NEEDED):
                points[j] = coordinatesArray[(i + j) % len(coordinatesArray)]
                indexes[j] = ((i + j) % len(coordinatesArray))

            areNotFourCollinear = True

            # Check collinearity of all possible combinations
            for k in range(0, NUM_POINTS_NEEDED):
                areNotFourCollinear &= not self.__are_three_points_collinear(points[k],
                                                                             points[(k + 1) % NUM_POINTS_NEEDED],
                                                                             points[(k + 2) % NUM_POINTS_NEEDED])

                # If points are collinear, stop looping
                if (areNotFourCollinear == False):
                    break

            # If all four points are non-collinear, return this combination of points
            if areNotFourCollinear:
                self.__logger.debug("geolocation/get_non_collinear_points: Returned " + str(points))

                # Sort and return the indexes in ascending order
                indexes.sort()
                return indexes
        
        self.__logger.debug("geolocation/get_non_collinear_points: Returned np.empty(shape=(0,2))")
        return np.empty(0)

    def calculate_pixel_to_geo_mapping(self):
        """
        Outputs transform matrix for mapping pixels to geographical points

        Returns
        -------
        np.array(shape=(3,3))
        """
        self.__logger.debug("geolocation/calculate_pixel_to_geo_mapping: Started")

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
        matrixProduct = (mappedGeoMatrix.dot(sourcePixelMatrixInverse))
        self.__logger.debug("geolocation/calculate_pixel_to_geo_mapping: Returned " + str(matrixProduct))
        return matrixProduct

    def convert_input(self):
        """
        Converts telemtry data into workable data

        Returns
        -------
        list:
            list of numpy arrays that represent the o, c, u and v vectors
        """
        self.__logger.debug("geolocation/convert_input: Started")

        # get plane and camera rotation matrices
        planeRotation = self.__calculate_rotation_matrix(self.__eulerPlane)
        cameraRotation = self.__calculate_rotation_matrix(self.__eulerCamera)

        # get plane and camera compound rotation matrix
        # note: apply camera rotation and then plane rotation in that order
        compoundRotationMatrix = np.matmul(planeRotation, cameraRotation)

        # calculate gps module to camera offset
        gpsCameraOffset = np.subtract(self.__CAMERA_OFFSET, self.__GPS_OFFSET)

        # calculate o, c, u, v vectors
        o = self.__calculate_o_vector(compoundRotationMatrix, gpsCameraOffset)
        c = self.__calculate_c_vector(compoundRotationMatrix)
        u = self.__calculate_u_vector(compoundRotationMatrix)
        v = self.__calculate_v_vector(c, u)

        self.__logger.debug("geolocation/convert_input: Returned " + str((o, c, u, v)))
        return (o, c, u, v)

    def __calculate_o_vector(self, compoundRotationMatrix: np.ndarray, gpsCameraOffset: np.ndarray) -> np.ndarray:
        """
        Returns a numpy array that contains the components of the o vector

        Parameters
        ----------
        compoundRotationMatrix: numpy array
            Array containing rotation matrix for camera and plane rotations
        gpsCameraOffset: numpy array
            The offset of the GPS to the camera on the plane in plane space

        Returns
        -------
        oVector: numpy array
            Array containing components of the o vector
        """
        self.__logger.debug("geolocation/__calculate_o_vector: Started")

        # get GPS module coordinates
        gpsModule = np.empty(3)
        gpsModule[0] = self.__longitude - self.__WORLD_ORIGIN[0]
        gpsModule[1] = self.__latitude - self.__WORLD_ORIGIN[1]
        gpsModule[2] = self.__altitude - self.__WORLD_ORIGIN[2]

        # transform gps-to-camera offset from plane space to world space by applying inverse rotation matrix
        # note: for rotation matrices, transpose is equivalent to inverse
        transposedMatrix = np.transpose(compoundRotationMatrix)
        rotatedOffset = np.matmul(transposedMatrix, gpsCameraOffset)

        # get camera coordinates in world space
        cameraCoordinates = np.add(gpsModule, rotatedOffset)
        self.__logger.debug("geolocation/__calculate_o_vector: Returned " + str(cameraCoordinates))
        return cameraCoordinates

    def __calculate_c_vector(self, compoundRotationMatrix: np.ndarray) -> np.ndarray:

        """
        Returns a numpy array that contains the components of the c vector

        Parameters
        ----------
        compoundRotationMatrix: numpy array
            Array containing rotation matrix for camera and plane rotations

        Returns
        -------
        cVector:
            Array containing components of the c vector
        """
        self.__logger.debug("geolocation/__calculate_c_vector: Started")

        # apply compound rotation matrix to cVectorCameraSpace
        cVector = np.matmul(compoundRotationMatrix, self.__C_VECTOR_CAMERA_SPACE)

        # normalize so that ||cVector|| = 1
        norm = np.linalg.norm(cVector)
        if (norm != 0):
            cVector = cVector / norm

        # reshape vector
        # in order to get an fov, fix the c vector and scale u and v with fov factors to get an fov
        squeezedVector = np.squeeze(cVector)
        self.__logger.debug("geolocation/__calculate_c_vector: Returned " + str(squeezedVector))
        return squeezedVector

    def __calculate_u_vector(self, compoundRotationMatrix: np.ndarray) -> np.ndarray:

        """
        Returns a numpy array that contains the components of the u vector (one of the camera rotation vectors)

        Parameters
        ----------
        compoundRotationMatrix: numpy array
            Array containing rotation matrix for camera and plane rotations
            
        Returns
        -------
        uVector: numpy array
            Array containing camera rotation vector
        """
        self.__logger.debug("geolocation/__calculate_u_vector: Started")

        # apply compound rotation matrix to uVectorCameraSpace
        uVector = np.matmul(compoundRotationMatrix, self.__U_VECTOR_CAMERA_SPACE)

        # normalize uVector so ||uVector|| = 1
        norm = np.linalg.norm(uVector)
        if (norm != 0):
            uVector = uVector / norm

        # reshape vector and apply field of view factor
        uVector = np.squeeze(uVector)
        scaledVector = uVector * self.__FOV_FACTOR_H
        self.__logger.debug("geolocation/__calculate_u_vector: Returned " + str(scaledVector))
        return scaledVector


    def __calculate_v_vector(self, c: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Returns a numpy array that contains the components of the v vector (remaining camera rotation vector)

        Parameters
        ----------
        c: numpy array
            Array containing camera direction vector
        u: numpy array
            Array containing one camera rotation vector

        Returns
        -------
        numpyArray:
            array containing the remaining camera rotation vector
        """
        self.__logger.debug("geolocation/__calculate_v_vector: Started")

        # cross product c and u to get v
        vVector = np.cross(c, u)

        # normalize vVector so ||vVector|| = 1
        norm = np.linalg.norm(vVector)
        if (norm != 0):
            vVector = vVector / norm

        # apply field of view factor
        vVector = np.squeeze(vVector)
        scaledVector = vVector * self.__FOV_FACTOR_V
        self.__logger.debug("geolocation/__calculate_v_vector: Returned " + str(scaledVector))
        return scaledVector

    def __calculate_rotation_matrix(self, eulerAngles: np.ndarray) -> np.ndarray:
        """
        Calculate and return rotation matrix given euler angles

        Parameters
        ----------
        eulerAngles: dict
            Dictionary containing euler angles with roll ('x'), pitch ('y') and yaw ('z') rotations in radians

        Returns
        -------
        rotationMatrix: numpy array
            Rotation matrix calculated from the given euler angles
        """
        self.__logger.debug("geolocation/__calculate_rotation_matrix: Started")

        # get euler angles
        # note: naming conventions specified in CV-Telemetry docs and commandModule specs
        yawAngle = eulerAngles["yaw"]
        pitchAngle = eulerAngles["pitch"]
        rollAngle = eulerAngles["roll"]

        # get plane yaw rotation matrix
        yawRotation = np.array([[float(np.cos([yawAngle])), -1 * float(np.sin([yawAngle])), 0],
                                [float(np.sin([yawAngle])), float(np.cos([yawAngle])), 0],
                                [0, 0, 1]])

        # get plane pitch rotation matrix
        pitchRotation = np.array([[float(np.cos([pitchAngle])), 0, float(np.sin([pitchAngle]))],
                                  [0, 1, 0],
                                  [-1 * float(np.sin([pitchAngle])), 0, float(np.cos([pitchAngle]))]])

        # get plane roll rotation matrix
        rollRotation = np.array([[1, 0, 0],
                                 [0, float(np.cos([rollAngle])), -1 * float(np.sin([rollAngle]))],
                                 [0, float(np.sin([rollAngle])), float(np.cos([rollAngle]))]])

        # get total plane rotation matrix based on euler rotation theorem
        # note: assume ZYX euler angles (i.e. R = RzRyRx); otherwise, change the multiplication order
        rotationMatrix = np.matmul(yawRotation, np.matmul(pitchRotation, rollRotation))

        self.__logger.debug("geolocation/__calculate_rotation_matrix: Returned " + str(rotationMatrix))
        return rotationMatrix

    # Private function that checks if the trimmed array is empty ie all values were too far apart from the median
    # Returns median if array is empty, otherwise finds the average of the trimmed array

    def __get_average_otherwise_median(self,trimmedArray,median):
        self.__logger.debug("geolocation/__get_average_otherwise_median: Started")

        if np.size(trimmedArray,0)==0:
            self.__logger.debug("geolocation/__get_average_otherwise_median: Returned " + str(median))
            return median
        else:
            average = np.average(trimmedArray,axis=0)
            self.__logger.debug("geolocation/__get_average_otherwise_median: Returned " + str(average))
            return average

    # Helper function required for input
    def concatenate_locations(self, newLocations):
        self.__logger.debug("geolocation/concatenate_locations: Started, input: " + str(newLocations))
        for i in range (0, len(newLocations)):
            self.__locationsList.append(newLocations[i])
        self.__logger.debug("geolocation/concatenate_locations: Finished")

    def get_best_location(self,inputLocationTupleList):
        self.__logger.debug("geolocation/get_best_location: Started")

        # For the  case of a single row matrix being passed to the function
        if np.size(inputLocationTupleList, 0) == 1:

            averagePair = np.vstack(inputLocationTupleList[:, 0]).astype(np.float64)
            averageError = np.vstack(inputLocationTupleList[:, 1]).astype(np.float64)


        else:

            # Splits the 3D numpy arrray into three separate arrays
            coordPair = np.vstack(inputLocationTupleList[:, 0]).astype(np.float64)
            errorArray = np.vstack(inputLocationTupleList[:, 1]).astype(np.float64)
            # confidenceArray = np.vstack(inputLocationTupleList[:, 2]).astype(np.float64)

            # Splits coordinate pair into x-coord and y-coord to remove outliers
            xCoord = np.vstack(coordPair[:, [0][0]]).astype(np.float64)
            yCoord = np.vstack(coordPair[:, [1][0]]).astype(np.float64)
            xCoordTrimmed = []
            yCoordTrimmed = []

            xCoordMedian = np.median(xCoord)
            yCoordMedian = np.median(yCoord)
            errorMedian = np.median(errorArray)

            # A modified version of the trimmed mean calculation. Here the distance measurement is distance between every x,y point from the median coordinate pair, x median and y median.
            # 7 is the distance for how far a given point can be from the median
            # TODO: Z-scores could be used instead
            for (x, y) in zip(xCoord, yCoord):
                distance = math.hypot(xCoordMedian - x, yCoordMedian - y)

                if distance < 7:
                    xCoordTrimmed.append(x)
                    yCoordTrimmed.append(y)

            # In the case of the values in the error array, the distance here is 3.5.
            errorArray = errorArray[(errorArray - errorMedian > -3.5) & (errorArray - errorMedian < 3.5)]

            averageX = self.__get_average_otherwise_median(xCoordTrimmed, xCoordMedian)
            averageY = self.__get_average_otherwise_median(yCoordTrimmed, yCoordMedian)
            averageError = self.__get_average_otherwise_median(errorArray, errorMedian)

            averagePair = (averageX, averageY)

        self.__logger.debug("geolocation/get_best_location: Returned " + str((averagePair, averageError)))
        return (averagePair,averageError)

    def run_locator(self, telemetry, coordinates):
        euler_angles_plane = telemetry["eulerAnglesOfPlane"]
        euler_angles_camera = telemetry["eulerAnglesOfCamera"]
        # Pls confirm shape of gpsCoordinates from command
        gpsLongitude = telemetry["gpsCoordinates"]["longitude"]
        gpsLatitude = telemetry["gpsCoordinates"]["latitude"]
        altitude = telemetry["gpsCoordinates"]["altitude"]

        # Expect euler angles to be in degrees
        self.__eulerCamera = self.__deg_vals_to_rad(euler_angles_camera)
        self.__eulerPlane = self.__deg_vals_to_rad(euler_angles_plane)

        # Competition
        # TODO Properly integrate lat-lon converters - refactor unit tests
        localCoordinates = self.local_from_lat_lon(gpsLatitude, gpsLongitude)
        self.__longitude = localCoordinates[0]
        self.__latitude = localCoordinates[1]
        self.__altitude = altitude

        camera_o, camera_c, camera_u, camera_v = self.convert_input()
        self.__cameraOrigin3o = camera_o
        self.__cameraDirection3c = camera_c
        self.__cameraOrientation3u = camera_u
        self.__cameraOrientation3v = camera_v

        point_pairs = self.gather_point_pairs()
        # If insufficient point pairs, exit this run and try again
        if len(point_pairs) < 4:
            return False, None

        # Slice point_pairs to shape (n, 2) for input of get_non_collinear_points
        points = point_pairs[:,1]

        # Get the 4 non-collinear indexes of the array above
        indexes = self.get_non_collinear_points(points)

        # If insufficient point pairs, exit this run and try again
        if len(indexes) < 4:
            return False, None

        # Create a subset of the (n, 2, 2) array above using the array of indexes
        non_collinear_points = (point_pairs[indexes])
        # non_collinear_points only stores the 4 non-collinear point pairs
        # indicated by the indexes array

        self.__pixelToGeoPairs = non_collinear_points
        tranformation_matrix = self.calculate_pixel_to_geo_mapping()

        local_coordinates = self.map_location_from_pixel(tranformation_matrix, coordinates)
        # [[ 1.0012e+07  -1.378e+07]
        # [ 1.0012e+07  -1.378e+07]]
        # ... (# of rows = # of points in parameter)

        # Competition
        # TODO Properly integrate lat-lon converters - refactor unit tests
        geo_coordinates = np.empty(shape=(len(coordinates), 2))
        for i in range (0, len(coordinates)):
            geo_coordinates[i] = (self.lat_lon_from_local(local_coordinates[i][0], local_coordinates[i][1]))
            
        return True, geo_coordinates

    @staticmethod
    def __deg_vals_to_rad(convert_dict):
        return dict(zip(convert_dict.keys(), list(map(lambda s: math.radians(s), convert_dict.values()))))

    def run_output(self, newLocations):
        self.__logger.debug("geolocation/run_output: Started")

        self.concatenate_locations(newLocations)
        locations = np.array(self.__locationsList, dtype=object)
        # [[-80.54561231850593 43.472406971594125]
        #  [-80.54561117040404 43.47240788947427]
        #  [-80.54561031256442 43.47240983973664]
        #  [-80.54560801000646 43.47241168180702]]
        # ... (# of points passed in)
        
        self.__logger.debug("geolocation/run_output: Returned " + str((True, self.get_best_location)))
        return True, self.get_best_location(locations)

    def map_location_from_pixel(self, transformationMatrix, pixels):
        """
        Maps Geographical Location Coordinates in the destination image
        
        Parameters
        -------
            transformationMatrix : np.array(shape=(3,3))
            pixels : np.array(shape=(5,2))

        Returns
        -------
            np.array(shape=(5,2))
        """
        self.__logger.debug("geolocation/map_location_from_pixel: Started")

        # Express all 2D coordinates of pixels as 3D coordinates with z value = 1
        pixels = np.insert(pixels, 2, 1, axis=1)

        # Compute Homogeneous Coordinates: Product of Image Pixels and Coordinates
        homogeneousCoordinates = np.matmul(transformationMatrix, pixels.T).T

        geoCoordinates = np.empty(shape=(0, 2))

        # Cycle through all homogenized coordinates of pixels
        for h in homogeneousCoordinates:
            # Checking if the homogenized value of Z equals 0. If so, we return an empty array.
            if np.allclose(h[2], 0):
                geoCoordinates = np.vstack((geoCoordinates, np.full((2), np.inf)))
                continue

            # Dehomogenizing the coordinate vector to compute the position in the destination image
            dehomogenizedX = h[0] / h[2]
            dehomogenizedY = h[1] / h[2]

            geoCoordinates = np.vstack((geoCoordinates, np.array([dehomogenizedX, dehomogenizedY])))
        self.__logger.debug("geolocation/map_location_from_pixel: Returned " + str(geoCoordinates))

        return geoCoordinates

    def write_locations(self, locations):
        with open('write.txt', 'a') as f:
            f.write('\n'.join([','.join(['{:4}'.format(item) for item in row]) for row in locations]))
            f.write('\n')