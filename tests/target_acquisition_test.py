import pytest
import cv2
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.mergeImageWithTelemetry.mergedData import MergedData

def testTargetAcquisition():
    tracker = TargetAcquisition()
    tracker.set_curr_frame(MergedData(cv2.imread('tests/testImages/frame0.jpg'), {}))
    assert tracker.get_coordinates() != None
