
THRESHHOLD = 30

class FramePreProc:

    """

        Filters out useless telemetry data i.e data whose change in euler angles from the previous state exceeds 30 degrees

        Attributes
        ----------
        
        eulerDictLast : dict
            dictionary holding the previous eulerAnglesOfCamera to compare to

    """

    def __init__(self, eulerDictLast):

        
        self.eulerDictLast = eulerDictLast

    def filter(self, eulerDict):

        """

        Finds the difference between present euler angles and those of the previous state.

        Attributes
        ----------

        eulerDict: dict
            dictionary holding present eulerAnglesOfCamera

        Returns
        -------
        Boolean: 'True' if absolute difference per angle does not exceed 30 degrees else returns 'False'

        """

        if self.eulerDictLast == None:
            return False

        yaw = abs(self.eulerDictLast['yaw'] - eulerDict['yaw'])
        pitch = abs(self.eulerDictLast['pitch'] - eulerDict['pitch'])
        roll = abs(self.eulerDictLast['roll'] - eulerDict['roll'])
            
        if (yaw >= THRESHHOLD or pitch >= THRESHHOLD or roll >= THRESHHOLD):
            return False
            
        else:
            return True 

    def update_last_dict(self, eulerDict):

        """

        Updates eulerDictLast to hold new values

        """

        self.eulerDictLast = eulerDict




