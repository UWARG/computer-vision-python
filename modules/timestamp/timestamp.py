import datetime
import numpy.typing as npt

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
    def __init__(self, data: npt.ArrayLike, timestamp: datetime.datetime = datetime.datetime.now()):
        self.data = data
        self.timestamp = timestamp
