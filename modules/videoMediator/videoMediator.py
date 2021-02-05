import TargetAcquisition
import DecklinkSRC

class VideoMediator:
    tentCoordinates = None
    decklinkSrc=DecLinkSRC()
    targetAcquisition=TargetAcquisition()

    def __init__(self):
        self.__currentFrame=None

        """
        initialize thread

        """


    def getTargets(self):
        current_frame=decklinkSrc.grab()
        tentCoordinates=targetAcquisition.get_coordinates(currentFrame)
