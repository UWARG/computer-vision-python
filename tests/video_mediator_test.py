import pytest
from modules.videoMediator.videoMediator import VideoMediator
from time import sleep


def testVideoMediator():
    videoMediator = VideoMediator(testMode=True)
    assert videoMediator.get_coordinates() == [False]
