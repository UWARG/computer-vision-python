"""
Geolocation module to map pixel coordinates to geographical coordinates
"""

import numpy as np


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


    def convert_input(self):
        """
        Converts telemtry data into workable data

        Returns
        -------
        list:
            list of numpy arrays that represent the o, c, u and v vectors
        """
        # get plane and camera rotation matrices
        planeRotation = self.__calculate_rotation_matrix(self.__eulerPlane)
        cameraRotation = self.__calculate_rotation_matrix(self.__eulerCamera)

        # get plane and camera compound rotation matrix
        # note: assume apply plane -> camera rotations in that order
        compoundRotationMatrix = np.matmul(cameraRotation, planeRotation)

        # calculate gps module to camera offset
        gpsCameraOffset = np.subtract(self.__CAMERA_OFFSET, self.__GPS_OFFSET)

        # calculate o, c, u, v vectors
        o = self.__calculate_o_vector(compoundRotationMatrix, gpsCameraOffset)
        c = self.__calculate_c_vector(compoundRotationMatrix)
        u = self.__calculate_u_vector(compoundRotationMatrix)
        v = self.__calculate_v_vector(c, u)

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
        return np.add(gpsModule, rotatedOffset)

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
        # apply compound rotation matrix to cVectorCameraSpace
        cVector = np.matmul(compoundRotationMatrix, self.__C_VECTOR_CAMERA_SPACE)

        # normalize so that ||cVector|| = 1
        norm = np.linalg.norm(cVector)
        if (norm != 0):
            cVector = cVector / norm

        # reshape vector
        return np.squeeze(cVector)

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
        # apply compound rotation matrix to uVectorCameraSpace
        uVector = np.matmul(compoundRotationMatrix, self.__U_VECTOR_CAMERA_SPACE)

        # normalize uVector so ||uVector|| = 1
        norm = np.linalg.norm(uVector)
        if (norm != 0):
            uVector = uVector / norm

        # reshape vector and apply field of view factor
        uVector = np.squeeze(uVector)
        return uVector * self.__FOV_FACTOR_H

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
        # cross product c and u to get v
        vVector = np.cross(c, u)

        # normalize vVector so ||vVector|| = 1
        norm = np.linalg.norm(vVector)
        if (norm != 0):
            vVector = vVector / norm

        # apply field of view factor
        vVector = np.squeeze(vVector)
        return vVector * self.__FOV_FACTOR_V

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
        # get euler angles
        # note: naming conventions specified in CV-Telemetry docs and commandModule specs
        yawAngle = eulerAngles["z"]
        pitchAngle = eulerAngles["y"]
        rollAngle = eulerAngles["x"]

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

        return rotationMatrix
