"""
Geolocation module to map pixel coordinates to geographical coordinates
"""

import numpy as np


class Geolocation:
    """
    Locates the geographical position of a set of pixels
    """

    """
    INPUT CONVERSION METHODS TO RETURN O, C and U/V vectors
    """

    """

        Methods
        ________

        __get_o_vector(latitude: int, longitude: int, altitude: int) -> dict?

        __get_c_vector(latitude: int, longitude: int, altitude: int, eulerCamera: dict, eulerPlane: dict) -> dict?

        __get_u_vector(latitude: int, longitude: int, altitude: int, eulerCamera: dict, eulerPlane: dict) -> dict?

        __get_v_vector(latitude: int, longitude: int, altitude: int, eulerCamera: dict, eulerPlane: dict) -> dict?
        
        convert_input()

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

    def convert_input(self, latitude: int, longitude: int, altitude: int, worldOrigin: np.ndarray, gpsOffset: np.ndarray, cameraOffset: np.ndarray, eulerCamera: dict, eulerPlane: dict):
        """
        Converts telemtry data into workable data

        Parameters
        ----------
        latitude: int
            GPS latitude of the plane
        longitude: int
            GPS longitude of the plane
        altitude: int
            GPS altitude of the plane
        worldOrigin: ndarray
            Coordinates of the origin of the world space
        gpsOffset: ndarray
            Offset from GPS module to origin of plane space
        cameraOffset: ndarray
            Offset from origin of plane space to camera
        eulerCamera : dict
            dictionary containing euler angles for the camera:
                - roll, pitch and yaw rotations with the keys 'x', 'y' and 'z' respectively
        eulerPlane : dict
            dictionary containing euler angles for the plane:
                - roll, pitch and yaw rotations with the keys 'x', 'y' and 'z' respectively
        Returns
        -------
        list:
            list of numpy arrays that represent the o, c, u and v vectors
        """

        # get plane and camera rotation matrices
        planeRotation = self.__calculate_rotation_matrix(eulerPlane)
        cameraRotation = self.__calculate_rotation_matrix(eulerCamera)

        # calculate gps module to camera offset
        gpsCameraOffset = np.add(gpsOffset, cameraOffset)

        # calculate o, c, u, v vectors
        o = self.__calculate_o_vector(latitude, longitude, altitude, gpsCameraOffset, cameraRotation, planeRotation, worldOrigin)
        c = self.__calculate_c_vector(cameraRotation, planeRotation)
        u = self.__calculate_u_vector(cameraRotation, planeRotation)
        v = self.__calculate_v_vector(c, u)

        return (o, c, u, v)

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

    def __calculate_o_vector(self, latitude: int, longitude: int, altitude: int, gpsCameraOffset:np.ndarray, cameraRotation: np.ndarray, planeRotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """
        Returns a numpy array that contains the components of the o vector

        Parameters
        ----------
        latitude: int
            The GPS latitude of the plane
        longitude: int
            The GPS longitude of the plane
        altitude: int
            The GPS altitude of the plane
        gpsCameraOffset: numpy array
            The offset of the GPS to the camera on the plane in plane space
        cameraRotation: numpy array
            Array containing rotation matrix for camera rotation
        planeRotation: numpy array
            Array containing rotation matrix for plane rotation
        origin:numpy array
            Array containing the origin of the world space

        Returns
        -------
        oVector: numpy array
            Array containing components of the o vector
        """
        # get GPS module coordinates
        gpsModule = np.zeroes(3)
        gpsModule[0] = origin[0] + latitude
        gpsModule[1] = origin[1] + altitude
        gpsModule[2] = origin[2] + longitude

        # transform gps-to-camera offset from plane space to world space by applying inverse
        # rotation matrix using (AB)^-1 = B^-1 A^-1
        rotationMatrix = np.matmul(cameraRotation, planeRotation)
        invertedMatrix = np.pinv(rotationMatrix)
        rotatedOffset = invertedMatrix.dot(offset)

        # get camera coordinates in world space
        oVector = np.add(rotatedOffset, gpsModule)

        return oVector

    def __calculate_c_vector(self, cameraRotation: np.ndarray, planeRotation: np.ndarray, cVectorCameraSpace=np.array([[1], [0], [0]])) -> np.ndarray:
        """
        Returns a numpy array that contains the components of the c vector

        Parameters
        ----------
        cameraRotation: numpy array
            Array containing rotation matrix for camera rotation
        planeRotation: numpy array
            Array containing rotation matrix for plane rotation
        cVectorCameraSpace: numpy array
            Array containing cVector in camera space (default value assumes cVector points along the roll axis)

        Returns
        -------
        cVector:
            Array containing components of the c vector
        """
        # apply plane rotation to camera direction
        cVector = planeRotation.dot(cVectorCameraSpace)

        # apply camera rotation
        # not: this assumes that camera euler angles are w.r.t. plane space
        cVector = cameraRotation.dot(cVector)

        # normalize since magnitude doesn't matter
        norm = np.linalg.norm(cVector)
        cVector = cVector / norm

        return cVector


    def __calculate_u_vector(self, cameraRotation: np.ndarray, planeRotation: np.ndarray, uVectorCameraSpace=np.array([[1], [0], [0]])) -> np.ndarray:
        """
        Returns a numpy array that contains the components of the u vector (one of the camera rotation vectors)

        Parameters
        ----------
        cameraRotation: numpy array
            Array containing rotation matrix for camera rotation
        planeRotation: numpy array
            Array containing rotation matrix for plane rotation
        uVectorCameraSpace: numpy array
            Array containing uVector in camera space (default value assumes uVector points along the pitch axis)
            
        Returns
        -------
        uVector: numpy array
            Array containing camera rotation vector
        """
        # apply plane rotation to camera direction
        uVector = planeRotation.dot(uVector)

        # apply camera rotation to camera direction
        # note: this assumes that camera euler angles are w.r.t. plane space
        uVector = cameraRotation.dot(uVector)

        # normalize output since camera direction magnitude doesn't matter
        norm = np.linalg.norm(uVector)
        uVector = uVector / norm

        return uVector

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
        return np.cross(c, u)

    def __calculate_rotation_matrix(self, eulerAngles: dict) -> np.ndarray:
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
        yawRotation = np.array([[1, 0, 0],
                                [0, float(np.cos([yawAngle])), -1 * float(np.sin([yawAngle]))],
                                [0, float(np.sin([yawAngle])), float(np.cos([yawAngle]))]])
        # get plane pitch rotation matrix
        pitchRotation = np.array([[float(np.cos([pitchAngle])), 0, float(np.sin([pitchAngle]))],
                                  [0, 1, 0],
                                  [-1 * float(np.sin([pitchAngle])), 0, float(np.cos([pitchAngle]))]])
        # get plane roll rotation matrix
        rollRotation = np.array([[float(np.cos([rollAngle])), -1 * float(np.sin([rollAngle])), 0],
                                 [float(np.sin([rollAngle])), float(np.cos([rollAngle])), 0],
                                 [0, 0, 1]])

        # get total plane rotation matrix based on euler rotation theorem
        # note: assume ZYX euler angles (i.e. R = RxRyRz); otherwise, change the multiplication order
        rotationMatrix = np.matmul(rollRotation, np.matmul(pitchRotation, yawRotation))

        return rotationMatrix