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
        dict<"angle" : float, "counterclockwise" : bool>
            Returns a dict with angle of rotation and whether rotation is clockwise or counterclockwise

                "angle" : float
                    Angle of rotation specified

                "counterclockwise" : bool
                    True if rotation should be counterclockwise
                    False if rotation should be clockwise

        """

        
        '''
            Calculating bearings between plane and tent
        '''
        planeLat = planeGPS["lat"]*math.pi/180
        planeLon = planeGPS["lon"]

        tentLat = tentGPS["lat"]*math.pi/180
        tentLon = tentGPS["lon"]

        diffLon = (tentLon - planeLon)*math.pi/180

        y = math.sin(diffLon)*math.cos(tentLat)
        x = math.cos(planeLat)*math.sin(tentLat) - math.sin(planeLat)*math.cos(tentLat)*math.cos(diffLon)

        theta = math.atan2(y,x)
        bearing = (theta*180/math.pi + 360)%360


        '''
            Calculating Angle of Rotation required by plane
        '''
        cc = False
        rotate = 0.0
        
        # Step 1: Calculate the difference in Bearings:
        diffBearing = bearing - angle

        # Step 2: Check for smallest rotation
        if diffBearing > 0:
            if diffBearing > 180:
                diffBearing = 360 - diffBearing
                cc = True
            
        
        elif diffBearing < 0:
            if diffBearing < -180:
                diffBearing = 360 + diffBearing
            else:
                diffBearing *= -1
                cc = True

        rotate = diffBearing

        return {"angle": rotate, "counterclockwise": cc}
