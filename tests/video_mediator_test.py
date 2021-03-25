import pytest
from modules.videoMediator.videoMediator import VideoMediator


# Integration test to check mediator behaviour
def testVideoMediator():
    videoMediator = VideoMediator(testMode=True)
    # By default if an empty frame is passed into the targetAcquisition class, "False" will be put into the pipeline
    assert videoMediator.get_coordinates() == [False]
