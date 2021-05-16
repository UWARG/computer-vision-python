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
    data : 
    timestamp : datetime.datetime

    Methods
    -------
    __init__(data, timestamp (optional, defaults to time of initialization)) 
        sets the frame and timestamp

    """
    def __init__(self, data: typing.Union[npt.ArrayLike, dict], timestamp: datetime.datetime = None):
        self.data = data
        self.timestamp = timestamp
        if timestamp is None: 
            self.timestamp = datetime.datetime.now()

