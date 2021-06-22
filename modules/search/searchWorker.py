from modules.search.Search import Search
import json
import logging

def searchWorker(plane_data, pylon_gps):
    """
    Search worker process to perform search
    Gets data from the input, performs search, and returns
    
    Parameters
    ----------
    plane_data: POGI object
    pylon_gps: dictionary containing last target gps coordinates, get from "temp_pylon_gps.json"
    
    Returns
    -------
    serach_result: Ground command object
    """
  
    logger = logging.getLogger()
    logger.debug("searchWorker: Start Search Module")
    
    search = Search()

    # Performing search using perform_search() of class Search
    plane_gps = {
        "lattitude": plane_data["gpsCoordinates"]["lattitude"],
        "longtitude": plane_data["gpsCoordinates"]["longtitude"]
    }
    search_result = search.perform_search(pylon_gps, plane_gps)
    
    logger.debug("searchWorker: Stop Search Module")

    # Putting data => a float value: search_result to output pipeline
    return search_result
