import pytest
import cv2
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
def testTargetAcquisition():
    tracker = TargetAcquisition()
    tracker.set_curr_frame(cv2.imread('tests/pylon_test.jpg'))
    assert tracker.get_coordinates() != None
    