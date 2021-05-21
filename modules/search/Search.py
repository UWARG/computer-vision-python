import math

class Search:
    """Performs search algorithm"""

    def __init__(self):
        pass

    def perform_search(self, tentGPS, planeGPS, angle):

        """
        Search program to determine the minimum rotation angle between heading and tent

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
        dict
            Returns a dictionary containing the angle between the plane and tent to be rotated (heading) and the distance to be travelled (latestDistance)
        """

        
        planeLat = planeGPS["lat"]*math.pi/180
        planeLon = planeGPS["lon"]

        tentLat = tentGPS["lat"]*math.pi/180
        tentLon = tentGPS["lon"]

        diffLon = (tentLon - planeLon)*math.pi/180

        y = math.sin(diffLon)*math.cos(tentLat)
        x = math.cos(planeLat)*math.sin(tentLat) - math.sin(planeLat)*math.cos(tentLat)*math.cos(diffLon)

        theta = math.atan2(y,x)
        bearing = (theta*180/math.pi + 360)%360

        return {"heading": bearing, "latestDistance": 0}
