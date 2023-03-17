import math
import logging
from decimal import Decimal

class Search:
    """Performs search algorithm"""

    def __init__(self):
        self.__logger = logging.getLogger()
        self.__logger.debug("Search/__init__: Started, Finished")

    def perform_search(self, tentGPS, planeGPS):

        """
        Search program to determine the minimum rotation angle between heading and tent

        Parameters
        ----------
        tentGPS : obj
          dictionary containing long and lat coordinates of the tent
        planeGPS : obj
          dictionary containing long and lat coordinate of the plane
        angle : float
          bearings notation in degrees

        Returns
        -------
        dict
            Returns a dictionary containing the angle between the plane and tent to be rotated (heading) and the distance to be travelled (latestDistance)
        """
        self.__logger.debug("Search/perform_search: Started")
        
        planeLon = Decimal(math.radians(planeGPS["longitude"]))
        planeLat = Decimal(math.radians(planeGPS["latitude"]))

        tentLon = Decimal(math.radians(tentGPS["longitude"]))
        tentLat = Decimal(math.radians(tentGPS["latitude"]))

        diffLon = tentLon - planeLon

        x = math.sin(diffLon)*math.cos(tentLat)
        y = math.cos(planeLat)*math.sin(tentLat)-math.sin(planeLat)*math.sin(tentLat)*math.cos(diffLon)
        b = math.atan2(x, y)
        b_deg = math.degrees(b)
        bearing = divmod(b_deg+360, 360)[1]

        self.__logger.debug("Search/perform_search: Returned " + str({"heading": bearing, "latestDistance": 0}))
        return {"heading": bearing, "latestDistance": 0}
