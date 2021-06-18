from modules.search.Search import Search
import json

def search_worker(pause, exitRequest, pipelineIn, pipelineOut):
    """
        Search worker process to perform search with multiprocessing
        Gets data from the input pipeline, performs search, and puts it into the output pipeline

        Parameters
        ----------
        pause : multiprocessing.Lock
          a lock shared between worker processes
        exitRequest : multiprocessing.Queue
          a queue that determines when the worker process stops
        pipelineIn : multiprocessing.Queue
          input pipeline
        pipelineOut : multiprocessing.Queue
          output pipeline

        Returns
        -------
        None
    """
    print("Start Search")

    search = Search()

    # Locking process
    pause.acquire()

    # Getting data => a dictionary with structure: {tentGPS: value, planeGPS: value, angle: value} from input pipeline
    plane_data = pipelineIn.get()
    with open("temp_pylon_gps", "r") as pylon_gps_file:
        pylon_gps = json.load(pylon_gps_file)

    # Performing search using perform_search() of class Search
    search_result = search.perform_search(pylon_gps, plane_data['planeGPS'], plane_data['angle'])

    # Putting data => a float value: search_result to output pipeline
    pipelineOut.put(search_result)

    # Releasing lock
    pause.release()
