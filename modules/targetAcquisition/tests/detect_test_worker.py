from modules.targetAcquisition.personDetection.detect import Detection
import cv2
import time

img1 = cv2.imread('modules/targetAcquisition/tests/testImages/frame0.jpg')
img2 = cv2.imread('modules/targetAcquisition/tests/testImages/frame1.jpg')
img3 = cv2.imread('modules/targetAcquisition/tests/testImages/frame2.jpg')
img4 = cv2.imread('modules/targetAcquisition/tests/testImages/frame3.jpg')
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


def imageFaker(pause, exitRequest, mergedDataPipelineIn):
    for image in images:
        time.sleep(2)
        mergedDataPipelineIn.put(image)
        print("put image")
    exitRequest.put("DONE")
    print("made exit request")

def logger(pause, exitRequest, coordinatesTelemetryPipelineIn):
    while True: 
        if not exitRequest.empty():
            break
        coordinates = coordinatesTelemetryPipelineIn.get()
        print(coordinates)