from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.timestamp.timestamp import Timestamp

import datetime
import numpy.typing as npt
import typing

class MergeImageWithTelemetry:
    """
    Merges image pipeline with telemetry pipeline
    ...

    Attributes
    ----------
    curData : Timestamp 
        the current telemetry data 
    start : bool
        whether curData is None or not (whether get_closest_telemetry has been called before)
    telemetryData : list<Timestamp>
        list of the current telemetry data and their times in the form of Timestamp objects

    Methods
    -------
    __init__()
        initiallizes attributes
    put_back(newTelemetryData: Timestamp)
        places given telemetry data at the back of the telemetryData list
    merge_with_closest_telemetry(imageTimestamp: datetime.datetime)
        finds the telemetry with timestamp closest to the input datetime
    """

    def __init__(self): 
        self.curData = None
        self.start = True
        self.telemetryData = []

    def put_back(self, newTelemetryData: Timestamp):
        self.telemetryData.append(newTelemetryData)

    def merge_with_closest_telemetry(self, imageTimestamp: datetime.datetime, imageData : npt.ArrayLike ): 
        """
        finds the telemetry with timestamp closest to the input datetime
        assumes that TelemetryData is sorted by increasing timestamp

        Parameters 
        ----------
        imageTimestamp : datetime.datetime 
            The time that the telemetry data should match with

        Returns 
        -------
        [success, telemetry]

        success : bool 
            whether such a telemetry timestamp could be found
        telemetry : Timestamp
            the closest time and the corresponding telemetry data in a Timestamp object
        """

        if len(self.telemetryData) == 0:
            if self.start: 
                return False, None
            return True, self.curData
        
        if self.start: 
            self.curData = self.telemetryData.pop(0)
            self.start = False
        
        timeDelta = abs(imageTimestamp - self.curData.timestamp)

        while True:
            if len(self.telemetryData) == 0:
                return True, self.curData
            
            nextTelemetry = self.telemetryData.pop(0)
            nextTimeDelta = abs(nextTelemetry.timestamp - imageTimestamp)
            
            if nextTimeDelta > timeDelta:
                ret = MergedData(imageData, self.curData.data)
                self.curData = nextTelemetry
                return True, ret
            else: 
                self.curData = nextTelemetry
                timeDelta = nextTimeDelta
