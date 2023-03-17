from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.timestamp.timestamp import Timestamp

import datetime
import numpy.typing as npt
import typing

class MergeImageWithTelemetry:
    """
    Merges image pipeline with telemetry pipeline

    The imports used by this class may require python 3.8 or greater
    ...

    Attributes
    ----------
    telemetryData : list<Timestamp>
        list of the current telemetry data and their times in the form of Timestamp objects
    image : Timestamp
        the current image to match telemetry to in the form of a Timestamp class

    Methods
    -------
    __init__()
        initiallizes attributes
    should_get_image() 
        whether a new image should be inputted to be matched 
            - old image has already matched a telemetry timestamp or
            - pipeline is just starting and there is no image yet
    set_image(image: Timestamp): 
        sets the input to be the current image
    put_back_telemetry(newTelemetryData: Timestamp)
        places given telemetry data at the back of the telemetryData list
    get_closest_telemetry()
        finds the telemetry with timestamp closest to the timestamp of the current image
    """

    def __init__(self):
        self.telemetryData = []
        self.image = None

    def should_get_image(self):
        return self.image == None

    def set_image(self, image: Timestamp):
        self.image = image

    def put_back_telemetry(self, newTelemetryData: Timestamp):
        self.telemetryData.append(newTelemetryData)

    def get_closest_telemetry(self):
        """
        finds the telemetry with closest timestamp to the image timestamp
        assumes that TelemetryData is sorted by increasing timestamp

        Returns
        -------
        [success, mergedData]

        success : bool
            whether such a telemetry timestamp could be found
        mergedData : MergedData
            the merged image and telemetry data whose time difference is the closest together
        """

        if len(self.telemetryData) == 0:
            return False, None
        if self.image == None:
            return False, None

        imageTimestamp = self.image.timestamp
        imageData = self.image.data

        oldestTelemetry = self.telemetryData.pop(0)
        timeDelta = abs(imageTimestamp - oldestTelemetry.timestamp)

        while True:
            # if we run out of telemetry data, unget the current telemetry and just return false 
            if len(self.telemetryData) == 0:
                self.telemetryData.append(oldestTelemetry)
                return False, None

            nextTelemetry = self.telemetryData.pop(0)
            nextTimeDelta = abs(nextTelemetry.timestamp - imageTimestamp)

            if nextTimeDelta > timeDelta:
                # un_get the next telemetry
                self.telemetryData.insert(0, nextTelemetry)
                self.image = None
                return True, MergedData(imageData, oldestTelemetry.data)
            else:
                oldestTelemetry = nextTelemetry
                timeDelta = nextTimeDelta
