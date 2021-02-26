from boxDetection.detect import yolov5_detect
from qrDetection.scan import qr_decoder

class Namespace:
    """
    Helper class to emulate CLI configs passed into YOLOv5 detect.py

    Attributes
    ----------
    __dict__ : dict
        Dict of parameters to pass into YOLOv5 detect.py
    qrMessage : string
        Text message decoded from a QR code

    Methods
    -------
    __init__(kwargs : dict)
    """

    def __init__(self, **kwargs):
        """
        Store list of keyword arguments

        Parameters
        ----------
        kwargs : dict
            Variable size dict containing parameters to pass into YOLOv5 detect.py
        """
        self.__dict__.update(kwargs)

YOLOV5_CONFIGS = Namespace(agnostic_nms=False, augment=False, classes=[0], conf_thres=0.4, 
                device='', exist_ok=False, img_size=416, iou_thres=0.45, 
                name='exp', project='runs/detect', save_conf=False, save_txt=False, 
                source='0', tfl_int8=False, update=False, view_img=False, 
                weights=['.\\boxDetection\\weights\\best.pb'])

class Taxi:
    """
    Performs cardboard box detection on a given video frame

    Attributes
    ----------
    boundingBoxes : list<tuple<tuple, tuple>>
        List of bounding boxes 

    Methods
    -------
    __init__()
        Initialize class variables
    __predict_boxes()
        Get bounding box of cardboard boxes from YOLOv5
    __read_qr(image : np.ndarray)
        Read message from QR code
    """
    
    def __init__(self):
        self.boundingBoxes = []

    def __predict_boxes(self):
        """
        PRIVATE: Get bounding box of cardboard boxes from YOLOv5
        """
        # TODO: Get detect.py to return coordinates of each cardboard box after predicting from one video frame
        # TODO: Decide on a return format:
            # Formats: [((x0, y0), (x1, y1))] or [((x, y), (w, h))]
        self.boundingBoxes = yolov5_detect(opt = YOLOV5_CONFIGS)

    def __read_qr(self, image):
        """
        PRIVATE: Read QR code from image

        Parameters
        ----------
        image : np.ndarray
            Image containing a single readable QR code
        """
        # TODO: Get qrDetection scan.py to return the QR message as string instead of displaying it
        self.qrMessage = qr_decoder(image)


# TODO: Set up video feed to feed image into __predict_boxes() and __read_qr()
'''
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    decoder(frame)
    cv2.imshow('Image', frame)
    code = cv2.waitKey(10)
    if code == ord('q'):
        break
'''