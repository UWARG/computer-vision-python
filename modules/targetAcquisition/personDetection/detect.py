import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, 'modules/targetAcquisition/Yolov5_DeepSort_Pytorch/yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.models.experimental import attempt_load
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.downloads import attempt_download
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.models.common import DetectMultiBackend
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages, LoadStreams
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path, set_logging)
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_sync
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.plots import Annotator, colors
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.deep_sort.utils.parser import get_config
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.deep_sort.deep_sort import DeepSort
from modules.targetAcquisition.Yolov5_DeepSort_Pytorch.yolov5.utils.augmentations import Albumentations, letterbox

class Detection:
    def __init__(self):
        self.weights='modules/targetAcquisition/best.pt'
        set_logging()
        self.device =  torch.device('cpu')
        self.cfg = get_config()
        self.cfg.merge_from_file('modules/targetAcquisition/personDetection/deep_sort.yaml')
        half = self.device.type != 'cpu'
        self.vidWriter = None
        self.tracker = DeepSort(self.cfg.DEEPSORT.MODEL_TYPE,
                        self.device,
                        max_dist=self.cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                        )

    def detect_boxes(self, current_frame):
        yolo_model = 'modules/targetAcquisition/best.pt'
        path = 'modules/targetAcquisition/personDetection'
        output = 'inference/output'
        imgsz = [640, 640]
        conf_thres = 0.3
        iou_thres = 0.5
        device = ''
        show_vid = True
        save_vid = True 
        save_txt = True
        classes = None
        agnostic_nms = False
        augment = False
        evaluate = False
        half = False
        visualize = False
        max_det = 1000
        dnn = False
        project = './runs/track'
        name = 'exp'
        exist_ok = True
        bs = 1
        frame_idx = 0
        

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

        # Set Dataloader
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()

        # extract what is in between the last '/' and last '.'
        txt_path = str(Path(save_dir)) + '/' + 'saved' + '.txt'

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        img0 = current_frame

        # Padding Resize
        img = letterbox(img0, [640, 640], 32, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im0s = img0
        s = f'image {frame_idx} {path}: ' #path is a mock of where the image would be if we were passing in a file

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # variable for return bounding box coordinates
        bbox_cord = []

        # Inference
        pred = model(img, augment, visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)
        print(pred)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), 0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                print(img.shape)
                print(im0.shape)
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                t4 = time_sync()
                outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # if outputs = 0, normalize xyxy and add to list

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        # normalize bbox xyxy and add tuples of bbox to list

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

            
            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


            # Save results (image with detections)
            if save_vid:
                if isinstance(self.vidWriter, cv2.VideoWriter):
                    self.vidWriter.write(im0)
                else:
                    self.vidWriter = cv2.VideoWriter('runs/track/exp/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (im0.shape[1], im0.shape[0]))
                    self.vidWriter.write(im0)
                print(save_path)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
            per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            print('Results saved to %s' % save_path)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

        # return list of bbox tuples

    def close_writer(self):
        self.vidWriter.release()