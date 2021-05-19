import datetime as datetime
import numpy.typing as npt
import typing

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

    """
    def __init__(self, data: typing.Union[npt.ArrayLike, dict], timestamp: datetime.datetime = None):
        self.data = data
        self.timestamp = timestamp
        if timestamp is None: 
            self.timestamp = datetime.datetime.now()