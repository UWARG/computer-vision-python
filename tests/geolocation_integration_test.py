import pytest
from modules.geolocation import geolocation


def test_run_locator():
    # This test data is bogus, pls get me some realistic data
    mock_camera_euler = {
        'x': 0,
        'y': -90,
        'z': 0
    }
    mock_plane_euler = {
        'x': 0,
        'y': 0,
        'z': 0
    }
    mock_gps = {
        "longitude": 0,
        "latitude": 0,
        "altitude": 100
    }
    mock_telemetry = {
        "eulerAnglesOfPlane": mock_plane_euler,
        "eulerAnglesOfCamera": mock_camera_euler,
        "gpsCoordinates": mock_gps
    }

    mock_coordinates = [[0, 0]]

    locator = geolocation.Geolocation()
    ret, data = locator.run_locator(mock_telemetry, mock_coordinates)

    assert ret is True
