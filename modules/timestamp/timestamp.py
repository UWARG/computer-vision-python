import datetime

class Timestamp: 
    """
    Adds the time stamp to a given input (either video frame or telemetry)

    ...
    Attributes
    ----------
    frame : 
    timestamp : datetime.datetime

    Methods
    -------
    __init__(frame, timestamp (optional, defaults to time of initialization)) 
        sets the frame and timestamp
    get_frame  
        returns the frame
    get_timestamp 
        returns the timestamp

    """
    def __init__(self, frame, timestamp = datetime.datetime.now()):
        self.frame  = frame
        self.timestamp = timestamp

    def get_frame(self):
        return self.frame

    def get_timestamp(self):
        return self.timestamp

