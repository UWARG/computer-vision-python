import pytest
import cv2
import time

from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.geolocation.geolocation import Geolocation

@pytest.fixture
def get_image():
    img1 = cv2.imread('frame0.jpg')
    return img1
    # cv2.imshow('img', img1) #JUST TO CHECK IF IMAGE IS TAKEN CORRECTLY
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test_targetAcquisition_to_geolocation(get_image):
        target = TargetAcquisition()
        location = Geolocation()

        while True:
            target.set_curr_frame(get_image)
            check1, coordinates_and_telemetry = target.get_coordinates(get_image)

            check2, geo_coordinates = location.run_locator() # put things in here parameter to test? find out 
            check3, locations = location.run_output()        # how to connect both modules through paramters?

            assert check1 == True and coordinates_and_telemetry != None
            assert check2 == True and geo_coordinates != None
            assert check3 == True and locations != None