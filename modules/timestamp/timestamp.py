from __future__ import annotations 

import datetime as datetime
import numpy.typing as npt
import typing
import time

class Timestamp: 
    """
    Adds the time stamp to a given input (either video frame or telemetry)

    ...
    Attributes
    ----------
    data : Union[npt.ArrayLike, dict]
    timestamp : datetime.datetime

    Methods
    -------
    __init__(data : np.ArrayLike, timestamp : datetime.datetime, optional) 
        sets the data and timestamp, timestamp defaults to time of initialization
    __get_average_time(self, otherTime : datetime.datetime)
        returns the average of the self.timestamp and the input datetime
    merge(self, other : Timestamp)
        merges two Timestamp objects in the following way:
            data is a tuple of both datas (self.data, other.data)
            timestamp is the average of the two times 

    """
    def __init__(self, data: typing.Union[npt.ArrayLike, dict], timestamp: datetime.datetime = None):
        self.data = data
        self.timestamp = timestamp
        if timestamp is None: 
            self.timestamp = datetime.datetime.now()

    def __get_average_time(self, otherTime: datetime.datetime) -> datetime.datetime:
        deltaOne = self.timestamp - datetime.datetime.fromtimestamp(0)
        deltaTwo = otherTime - datetime.datetime.fromtimestamp(0)

        averageDelta = (deltaOne + deltaTwo) / 2 
        averageTime = datetime.datetime.fromtimestamp(averageDelta / datetime.timedelta(seconds=1))
        return averageTime

    def merge(self, other: typing.type[Timestamp]):
        self.data = (self.data, other.data)
        self.timestamp = self.__get_average_time(other.timestamp)
