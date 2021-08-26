from modules.commandModule.commandWorker_flight import flight_command_worker
from modules.commandModule.tests.testCases.generate_temp_json import generate_temp_json
import json
import multiprocessing as mp
import os

PIGO_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPigo.json")
POGI_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "testPogi.json")
generated_pigo = generate_temp_json(PIGO_DIR)
generated_pogi = generate_temp_json(POGI_DIR)

pipeline_in = mp.Queue()
pipeline_out = mp.Queue()
pause = mp.Lock()
quit = mp.Queue()

expected_value = {'latitude': 1.2, 'longitude': 3.4, 'altitude': 5.6}

pipeline_in.put([expected_value, [0]])

flight_command_worker(pause, quit, pipeline_in, pipeline_out, PIGO_DIR, POGI_DIR)


with open(PIGO_DIR) as pigo:
    data = json.load(pigo)
    assert data['gpsCoordinates']['latitude'] == 1.2
    assert data['gpsCoordinates']['longitude'] == 3.4
    assert data['gpsCoordinates']['altitude'] == 5.6

os.remove(generated_pogi)
os.remove(generated_pigo)