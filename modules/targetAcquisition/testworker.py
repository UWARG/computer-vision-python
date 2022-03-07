from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.targetAcquisition.personDetection.detect import Detection
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.newTrack import detect
import logging
import cv2
import time

class configObject: 
    yolo_model = 'modules/targetAcquisition/best.pt'
    deep_sort_model = 'osnet_x0_25'
    path = 'modules/targetAcquisition'
    output = 'inference/output'
    imgsz = [640, 640]
    conf_thres = 0.3
    iou_thres = 0.5
    fourcc = 'mp4v'
    device = ''
    show_vid = True
    save_vid = False 
    save_txt = True
    classes = None
    agnostic_nms = False
    augment = False
    evaluate = False
    config_deepsort = "modules/targetAcquisition/Yolov5_DeepSort_Pytorch/deep_sort/configs/deep_sort.yaml"
    half = False
    visualize = False
    max_det = 1000
    dnn = False
    project = './runs/track'
    name = 'exp'
    exist_ok = False

img1 = cv2.imread('modules/targetAcquisition/frame0.jpg')
img2 = cv2.imread('modules/targetAcquisition/frame1.jpg')
img3 = cv2.imread('modules/targetAcquisition/frame2.jpg')
img4 = cv2.imread('modules/targetAcquisition/frame3.jpg')
images = [img1, img2, img3, img4]


def targetAcquisitionWorker(pause, exitRequest, mergedDataPipelineIn):

    detector = Detection()
    
    while True:
        print(exitRequest.empty())
        if not exitRequest.empty():
            break
        
        pause.acquire()
        pause.release()

        curr_frame = mergedDataPipelineIn.get()
        print("got frame")

        if curr_frame is None:
            break
        
        # Set the current frame
        print("running detect")
        detector.detect_boxes(curr_frame)

        # Run model
        # res, coordinatesAndTelemetry = targetAcquisition.get_coordinates()
        # if not res:
        #     continue


def imageFaker(pause, exitRequest, mergedDataPipelineIn):
    for image in images:
        time.sleep(2)
        mergedDataPipelineIn.put(image)
        print("put image")
    exitRequest.put("DONE")
    print("made exit request")