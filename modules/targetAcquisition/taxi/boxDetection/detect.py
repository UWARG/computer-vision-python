import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from numpy import random
import numpy as np

from boxDetection.models.experimental import attempt_load
from boxDetection.utils.datasets import letterbox
from boxDetection.utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from boxDetection.utils.plots import plot_one_box
from boxDetection.utils.torch_utils import select_device, load_classifier, time_synchronized
import tensorflow as tf
from tensorflow import keras


class Detection:
        
    def __init__(self):
        self.weights=['./boxDetection/weights/best.pb']
        # Initialize
        set_logging()
        self.device =  torch.device('cpu')   # device = select_device(opt.device)
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        self.weights = self.weights[0] if isinstance(self.weights, list) else self.weights
        suffix = Path(self.weights).suffix
        
        with open('boxDetection/data/data.yaml') as f:
            self.names = yaml.load(f, Loader=yaml.FullLoader)['names']  # class names (assume COCO)

        self.backend = 'graph_def'

        # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
        # https://github.com/leimao/Frozen_Graph_TensorFlow
        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.compat.v1.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        graph = tf.Graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(open(self.weights, 'rb').read())
        frozen_func = wrap_frozen_graph(graph_def=graph_def, inputs="x:0", outputs="Identity:0")
        self.model = frozen_func
        
    def detect_boxes(self, img):
        
        #important ones
        conf_thres=0.4
        img_size=416
        iou_thres=0.45
        source='0'
        #less important ones
        agnostic_nms=False
        augment=False
        classes=[0]
        exist_ok=False
        name='exp'
        project='boxDetection/runs/detect'
        save_conf=False
        tfl_int8=False
        update=False
        #other variables
        imgsz = img_size
        
        
        # Set Dataloader
        vid_path, vid_writer = None, None
        
        # Get names and colors
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        t0 = time.time()
        img0 = img.copy()
        # Padded resize
        img = letterbox(img0, new_shape=img_size, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
            
        
        #variable for return bounding box coordinates
        bbox_cord = []
        
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(x=tf.constant(img.permute(0, 2, 3, 1).cpu().numpy())).numpy()
        
        # Denormalize xywh
        pred[..., :4] *= img_size
        pred = torch.tensor(pred)
            
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            im0 = img0
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {self.names[int(c)]}s, '  # add to string

                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bbox_cord.append(((int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3]))))
                

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
        print(f'Done. ({time.time() - t0:.3f}s)')
        return bbox_cord


