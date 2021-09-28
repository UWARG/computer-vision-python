
THRESHHOLD = 30

class FramePreProc:

    """

        Filters out useless telemetry data i.e data whose change in euler angles from the previous state exceeds 30 degrees

        Attributes
        ----------
        eulerDict : dict
            dictionary holding the eulerAnglesOfCamera data from telemetry
        eulerDictLast : dict
            dictionary holding the previous eulerAnglesOfCamera to compare to

    """

    def __init__(self, eulerDict, eulerDictLast):

        self.eulerDict = eulerDict
        self.eulerDictLast = eulerDictLast

    def filter(self):

        """

        Finds the difference between present euler angles and those of the previous state.

        Returns
        -------
        Boolean: 'True' if absolute difference per angle does not exceed 30 degrees else returns 'False'

        """

        yaw = abs(self.eulerDictLast['yaw'] - self.eulerDict['yaw'])
        pitch = abs(self.eulerDictLast['pitch'] - self.eulerDict['pitch'])
        roll = abs(self.eulerDictLast['roll'] - self.eulerDict['roll'])
            
        if (yaw >= THRESHHOLD or pitch >= THRESHHOLD or roll >= THRESHHOLD):
            return False
            
        else:
            return True 




