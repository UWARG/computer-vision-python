from numpy import outer
import pytest
import cv2
import os.path

from modules.decklinksrc.decklinkSrcWorker_taxi import DeckLinkSRC
from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.geolocation.geolocation import Geolocation

@pytest.fixture
def get_image():
    img1 = cv2.imread('tests/testImages/pylon_test.jpg')
    assert img1.any() != None
    # cv2.imshow('img', img1) # CHECK IF IMAGE IS TAKEN CORRECTLY
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    yield img1

def test_targetAcquisition_to_geolocation(get_image):
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
    # print(check1,coordinates_and_telemetry)

    location.set_constants()
    check2, geo_coordinates = location.run_locator(coordinates_and_telemetry[1], [[0, 0],[60, 523], [200,0], [430,505]])
    # connection between targetAcquisition and geolocation above: (target.telemetryData)

    # print (check2, geo_coordinates)
    # True    [[    -80.546      43.472]
    #         [    -80.546      43.472]
    #         [    -80.546      43.472]
    #         [    -80.546      43.472]]

    save_path = os.path.join(os.getcwd(), 'modules/mapLabelling')
    completeName = os.path.join(save_path, 'new.csv')
    location.write_locations(geo_coordinates, completeName)

    assert check1 == True 
    assert coordinates_and_telemetry != None

    assert check2 == True
    assert geo_coordinates is not None
