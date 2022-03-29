import pytest
from modules.targetAcquisition.personDetection.detect import Detection
import cv2
import time
import multiprocessing as mp
from modules.targetAcquisition.tests.detect_test_worker import targetAcquisitionWorker, imageFaker, logger

img1 = cv2.imread('tests/testImages/frame0.jpg')
img2 = cv2.imread('tests/testImages/frame1.jpg')
img3 = cv2.imread('tests/testImages/frame2.jpg')
img4 = cv2.imread('tests/testImages/frame3.jpg')
images = [img1, img2, img3, img4]


def targetAcquisitionWorker(pause, exitRequest, mergedDataPipelineIn, coordinatesTelemetryPipelineOut):

    detector = Detection()
    
    while True:
        if not exitRequest.empty():
            break
        
        pause.acquire()
        pause.release()

        curr_frame = mergedDataPipelineIn.get()
        print("got frame")

        if curr_frame is None:
            continue
        
        print("running detect")
        coordinates = detector.detect_boxes(curr_frame)
        coordinatesTelemetryPipelineOut.put(coordinates)


def imageFaker(exitRequest, mergedDataPipelineIn):
    for image in images:
        time.sleep(2)
        mergedDataPipelineIn.put(image)
        print("put image")
    exitRequest.put("DONE")
    print("made exit request")

def logger(exitRequest, coordinatesTelemetryPipelineIn):
    while True: 
        if not exitRequest.empty():
            break
        coordinates = coordinatesTelemetryPipelineIn.get()
        print(coordinates)
    
# Run this from main
def test(): 
    pipelineIn = mp.Queue()
    pipelineOut = mp.Queue()
    pause = mp.Lock()
    quit = mp.Queue()

    processes = [
        mp.Process(target = targetAcquisitionWorker, args = (pause, quit, pipelineIn, pipelineOut)),
        mp.Process(target = imageFaker, args = (quit, pipelineIn)),
        mp.Process(target = logger, args = (quit, pipelineOut)),
    ]

    for p in processes:
        print("starting proccess")
        p.start()
        
    for p in processes:
        p.join()