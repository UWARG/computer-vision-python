from modules.search.Search import Search
from modules.search.searchWorker import searchWorker

mock_tent = {
    "longitude": -114.114693,
    "latitude": 51.083665
}
mock_plane = {
    "longitude": -114.10714,
    "latitude": 51.059971
}


def test_search_function():
    search = Search()
    command = search.perform_search(tentGPS=mock_tent, planeGPS=mock_plane)
    assert 0.9 * 168 < command["heading"] < 1.1 * 168
    assert command["latestDistance"] == 0


def test_search_worker():
    mock_plane_data = {
        "gpsCoordinates": mock_plane
    }
    command = searchWorker(mock_plane_data, mock_tent)
    assert 0.9 * 168 < command["heading"] < 1.1 * 168
    assert command["latestDistance"] == 0
