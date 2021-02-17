import math
import numpy as np
from numpy import sin, cos


class Search:
    """Performs search algorithm"""

    def __init__(self):
        pass

    def perform_search(self, tentGPS, planeGPS, angle):
        """Search program to determine the minimum rotation angle between heading and tent

        Parameters
        ----------
        tentGPS : obj
          dictionary containing lat and long coordinates of the tent
        planeGPS : obj
          dictionary containing lat and long coordinate of the plane
        angle : float
          bearings notation in degrees

        Returns
        -------
        int
          returns the minimum rotation angle

        """
        # converts heading degrees into radians
        heading = math.radians(angle)
        # vector of the plane's current heading
        vHeading = np.array([cos(heading), sin(heading)])
        # vector from plane to tent
        vPlaneTent = np.array(
            [tentGPS["lat"] - planeGPS["lat"], tentGPS["lon"] - planeGPS["lon"]])

        # dot product of vectors
        dot = np.dot(vHeading, vPlaneTent)

        # find the angle between the two vectors
        rotation = math.acos(dot/(np.linalg.norm(vHeading)
                                  * np.linalg.norm(vPlaneTent)))
        print(math.degrees(rotation))
        return rotation
