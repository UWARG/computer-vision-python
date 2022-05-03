from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.geolocation.geolocation import Geolocation

if __name__ == "__main__":

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

    location.set_constants()
    check2, geo_coordinates = location.run_locator(mock_telemetry, [[0, 0],[100, 100], [200,300], [400,500]])
