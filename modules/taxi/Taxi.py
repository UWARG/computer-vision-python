from boxDetection.detect import detect


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt = Namespace(agnostic_nms=False, augment=False, classes=[0], conf_thres=0.4, 
                device='', exist_ok=False, img_size=416, iou_thres=0.45, 
                name='exp', project='runs/detect', save_conf=False, save_txt=False, 
                source='0', tfl_int8=False, update=False, view_img=False, 
                weights=['.\\boxDetection\\weights\\best.pb'])

detect(opt = opt)