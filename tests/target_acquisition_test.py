import pytest
import cv2
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.mergeImageWithTelemetry.mergedData import MergedData
import os 

os.chdir('modules/targetAcquisition/Yolov5_DeepSort_Pytorch')
os.system('git submodule update --init')
os.chdir('../../../')


def testTargetAcquisition():
    tracker = TargetAcquisition()
    tracker.set_curr_frame(MergedData(cv2.imread('tests/pylon_test.jpg'), {}))
    assert tracker.get_coordinates() != None
