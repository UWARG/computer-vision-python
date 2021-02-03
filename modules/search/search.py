import math
import numpy as np
from numpy import sin, cos


def altSearch(tent_GPS, plane_GPS, angle):
    """Search program to determine the minimum rotation angle between heading and tent

    Parameters
    ----------
    tent_GPS : obj
      dictionary containing lat and long coordinates of the tent
    plane_GPS : obj
      dictionary containing lat and long coordinate of the plane
    angle : in
      bearings notation in radians

    Returns
    -------
    int
      returns the minimum rotation angle

    """
    angle = math.radians(angle)
    # vector of the plane's current heading
    v_heading = np.array([cos(angle), sin(angle)])
    # vector from plane to tent
    v_plane_tent = np.array(
        [tent_GPS["lat"] - plane_GPS["lat"], tent_GPS["lon"] - plane_GPS["lon"]])

    # dot product of vectors
    dot = np.dot(v_heading, v_plane_tent)

    # find the angle between the two vectors
    rotation = math.acos(dot/(np.linalg.norm(v_heading)
                              * np.linalg.norm(v_plane_tent)))
    print(math.degrees(rotation))
    return rotation
