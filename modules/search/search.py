import geopy
from numpy import arctan2,random,sin,cos,degrees

def searchProgram(tent_GPS, plane_GPS, heading):
  '''
  tent_GPS and plane_GPS are dictionaries with lat and lon values
  '''
  
  '''
  Calculating Bearing between Tent and Plane
  '''
  # STEP 1: Calculating Difference in Longitude
  dL = tent_GPS.lon - plane_GPS.lon

  # STEP 2: Converting Longitude and Latitude to corresponding XY coordinates
  X = cos(tent_GPS.lat)* sin(dL)
  Y = cos(plane_GPS.lat)*sin(tent_GPS.lat) - sin(plane_GPS.lat)*cos(tent_GPS.lat)* cos(dL)

  # Step 3: Calculate Bearings in Radians and then converting to degrees
  bearing = (degrees(arctan2(X,Y))+360) % 360

  
  '''
  Calculating Angle of Rotation
  '''
  cc = False
  rotate = 0.0
  
  # Step 1: Calculate the difference in Bearings:
  dB = bearing - heading

  # Step 2: Check for smallest rotation
  if dB > 0:
    if dB > 180:
      dB = 360-dB
      cc = True
    rotate = dB
  
  elif dB < 0:
    if dB < -180:
      dB = 360+dB
    else:
      dB *= -1
      cc = True
    rotate = dB


''' REACHING HERE:
        rotate (float): stores angle of rotation
        cc (bool): stores if rotation should be counterclockwise (cc == True) or not
'''