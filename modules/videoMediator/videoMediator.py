import TargetAcquisition
import DecklinkSRC

class VideoMediator:
    tent_coordinates = None
    decklinksrc=DecLinkSRC()
    target_acquisition=TargetAcquisition()

    def __init__(self):
        self.__current_frame=None

        """
        initialize thread

        """


    def getTargets(self):
        current_frame=decklinksrc.grab()
        tent_coordinates=target_acquisition.get_coordinates(current_frame)
