import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0,'modules/targetAcquisition/Yolov5_DeepSort_Pytorch')
sys.path.insert(0,'modules/targetAcquisition/Yolov5_DeepSort_Pytorch/deep_sort/deep/reid')
sys.path.insert(0, 'modules/targetAcquisition/Yolov5_DeepSort_Pytorch/yolov5')

import os
import platform
import shutil
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.models.common import DetectMultiBackend
from modules.targetAcquisition.personDetection.detectdependencies import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, xywh2xyxy, increment_path, set_logging)
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_sync
from detectdependencies import Annotator, colors
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.deep_sort.utils.parser import get_config
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.deep_sort.deep_sort import DeepSort
from detectdependencies import Albumentations, letterbox

class Detection:
    def __init__(self):
        self.weights='modules/targetAcquisition/personDetection/weights/best.pt'
        set_logging()
        self.device =  torch.device('cpu')
        self.cfg = get_config()
        self.cfg.merge_from_file('modules/targetAcquisition/personDetection/config/deep_sort.yaml')
        half = self.device.type != 'cpu'
        self.vidWriter = None # used to save a video of the tracking
        self.tracker = DeepSort(self.cfg.DEEPSORT.MODEL_TYPE,
                        self.device,
                        max_dist=self.cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                        )

    def detect_boxes(self, current_frame):
        #for config
        yolo_model = 'modules/targetAcquisition/personDetection/weights/best.pt'
        path = 'modules/targetAcquisition/personDetection' # used in construction of imShow window name param
        output = 'inference/output'
        imgsz = 416
        conf_thres = 0.3
        iou_thres = 0.45
        device = ''

        # For visualizing and saving output
        show_vid = True # shows each image 
        save_vid = False 
        save_txt = False 
        classes = None
        agnostic_nms = False
        augment = False
        evaluate = False
        half = False
        visualize = False
        max_det = 1000
        dnn = False
        project = 'modules/targetAcquisition/personDetection/runs/track' #determines where results will be saved to (txt/video)
        name = 'exp'
        exist_ok = True
        bs = 1
        frame_idx = 0 # frame id

        if not evaluate:
            if os.path.exists(output):
                pass
                shutil.rmtree(output)  # delete output folder
            os.makedirs(output)  # make new output folder

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()

        # extract what is in between the last '/' and last '.'
        txt_path = str(Path(save_dir)) + '/' + 'results' + '.txt'

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        img0 = current_frame

        # Padding Resize
        img = letterbox(img0, imgsz, 32, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im0s = img0

        s = f'image {frame_idx} {path}: '

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # variable for returning bounding box coordinates
        bbox_cord = []

        # Inference
        pred = model(img, augment, visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), 0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # pass detections to DeepSort
                xywhs = xyxy2xywh(det[:, 0:4])
                t4 = time_sync()
                confs = det[:, 4]
                clss = det[:, 5]

                outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # if outputs = 0 still add to list 
                if len(outputs) == 0:
                    for *xyxy, conf, cls in det:
                        bbox_cord.append(((int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))))
                        bbox_cord.sort(key=lambda tup: tup[0][0])

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        # normalize bbox xyxy and add tuples of bbox to list
                        bbox_cord.append(((output[0],output[1]),(output[2],output[3])))
                        bbox_cord.sort(key=lambda tup: tup[0][0])
                            
                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                                                               
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
            else:
                self.tracker.increment_ages()
                LOGGER.info('No detections')

            frame_idx += 1

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


            # Save results (image with detections)
            if save_vid:
                if isinstance(self.vidWriter, cv2.VideoWriter): # checks if vidWriter already exists, if it does, continues using that one
                    self.vidWriter.write(im0)
                else: 
                    self.vidWriter = cv2.VideoWriter('runs/track/exp/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (im0.shape[1], im0.shape[0]))
                    self.vidWriter.write(im0)

        return bbox_cord

    # function to close vid_writer, should call this from worker before ending if we want to save a video of the tracking 
    def close_writer(self):
        self.vidWriter.release()
