from ...commandWorker_flight import flight_command_worker
import pytest
import json
import multiprocessing as mp

PIGO_DIR = "test_pigo.json"

pipeline_in = mp.Queue()
pipeline_out = mp.Queue()

expected_value = {'latitude': 1.2, 'longitude': 3.4, 'altitude': 5.6}

pipeline_in.put([expected_value, [0]])

flight_command_worker(pipeline_in, pipeline_out, PIGO_DIR)

with open(PIGO_DIR) as file:
    data = json.load(file)
    assert data['gpsCoordinates']['latitude'] == 1.2
    assert data['gpsCoordinates']['longitude'] == 3.4
    assert data['gpsCoordinates']['altitude'] == 5.6
