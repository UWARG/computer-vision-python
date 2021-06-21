from modules.search.Search import Search
import json
import logging

def searchWorker(plane_data):
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
  
    logger = logging.getLogger()
    logger.debug("searchWorker: Start Search Module")
    
    search = Search()
    
    # Getting data => a dictionary with structure: {tentGPS: value, planeGPS: value, angle: value} from input pipeline
    with open("temp_pylon_gps", "r") as pylon_gps_file:
        pylon_gps = json.load(pylon_gps_file)

    # Performing search using perform_search() of class Search
    plane_gps = {
        "lat": plane_data["gpsCoordinates"]["lattitude"],
        "lon": plane_data["gpsCoordinates"]["longtitude"]
    }
    search_result = search.perform_search(pylon_gps, plane_gps, plane_data['heading'])
    
    logger.debug("searchWorker: Stop Search Module")

    # Putting data => a float value: search_result to output pipeline
    return search_result
