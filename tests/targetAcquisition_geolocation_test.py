from numpy import outer
import pytest
import cv2
import time
import datetime as datetime
import os.path

from modules.decklinksrc.decklinkSrcWorker_taxi import DeckLinkSRC
from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.geolocation.geolocation import Geolocation
from modules.timestamp.timestamp import Timestamp
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry

# @pytest.fixture
def get_image():
    img1 = cv2.imread('frame1.jpg')
    return img1
    # cv2.imshow('img', img1) #JUST TO CHECK IF IMAGE IS TAKEN CORRECTLY
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test_targetAcquisition_to_geolocation(get_image):
    decklinkSrc = DeckLinkSRC()
    target = TargetAcquisition()
    location = Geolocation()

    mock_camera_euler = {
        'yaw': 5.0,
        'pitch': 60,
        'roll': 2
    }
    mock_plane_euler = {
        'yaw': 0,
        'pitch': 0,
        'roll': 0
    }
    mock_gps = {
        "longitude": -80.5449,
        "latitude": 43.4723,
        "altitude": 100
    }
    mock_telemetry = {
        "eulerAnglesOfPlane": mock_plane_euler,
        "eulerAnglesOfCamera": mock_camera_euler,
        "gpsCoordinates": mock_gps
    }
    merged = MergedData(get_image, mock_telemetry)

    target.set_curr_frame(merged)
    check1, coordinates_and_telemetry = target.get_coordinates()

    location.set_constants()
    check2, geo_coordinates = location.run_locator(merged.telemetry, [[0, 0],[60, 523], [200,0], [430,505]])
    # check3, locations = location.run_output(geo_coordinates) 
    #print (check1, coordinates_and_telemetry)
    print (check2, geo_coordinates)
    # True    [[    -80.546      43.472]
    #         [    -80.546      43.472]
    #         [    -80.546      43.472]
    #         [    -80.546      43.472]]

    save_path = save_path = os.path.join(os.getcwd(), 'modules/mapLabelling')
    completeName = os.path.join(save_path, 'new.csv')
    location.write_locations(geo_coordinates, completeName)

    # print (check3, locations)

    # assert check1 == True 
    # assert coordinates_and_telemetry != None

    # assert check2 == True
    # assert geo_coordinates != None

    # assert check3 == True 
    # assert locations != None

if __name__ == "__main__":
    test = get_image()
    test_targetAcquisition_to_geolocation(test)
