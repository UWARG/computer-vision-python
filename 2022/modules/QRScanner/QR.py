from ast import Constant
import logging
from pickle import NONE
import cv2

from pyzbar import pyzbar

class QRScanner:
    """
    Module encompassing logic associated with QR code scanning

    ...

    Attributes
    ----------
    codes : list<dict<str: str, str: tuple>>
        List of decoded QR codes in format {"text": given text, "rect": (x, y, w, h)},
        where x, y identify top left corner of bounding box & w, h identify width and height
    __logger : Logger
        Program-wide logger
    
    Methods
    -------
    __init__()
        Intitializes codes to an empty list
    main(frame : np.ndarray)
        Runs get_qr_codes to populate self.codes & draw_qr_codes to draw bounding boxes
    get_qr_codes(frame : np.ndarray)
        Uses pyzbar to find QR codes within an image, populates self.codes
    draw_qr_codes(frame : np.ndarray)
        Uses opencv to draw bounding boxes within the frame, returns the frame
    """
    def __init__(self):
        """
        Initializes self.codes
        """
        self.__logger = logging.getLogger()
        self.__logger.debug("QR/__init__: Started")

        self.codes = []

        self.__logger.debug("QR/__init__: Finished")

    def main(self, frame):
        """
        Runs get_qr_codes to populate self.codes & draw_qr_codes to draw bounding boxes

        Parameters
        ----------
        frame : np.ndarray
            Image to be processed
        """
        self.__logger.debug("QR/main: Started")

        self.codes = []
        self.get_qr_codes(frame)

        self.__logger.debug("QR/main: Finished")
        return self.draw_qr_codes(frame)

    def get_qr_codes(self, frame):
        """
        Uses pyzbar to find QR codes within an image, populates self.codes

        Parameters
        ----------
        frame : np.ndarray
            Image to be processed
        """
        qrCodes = pyzbar.decode(frame)
        
        for qr in qrCodes:
            (x, y, w, h) = qr.rect
            
            text = qr.data.decode("utf-8")
            self.codes.append({"text": text, "rect": (x, y, w, h)})
        return frame

    def draw_qr_codes(self, frame):
        """
        Uses opencv to draw bounding boxes within the frame, returns the frame
        
        Parameters
        ----------
        frame : np.ndarray
            Image to be processed

        Returns
        -------
        np.ndarray
            Returns the frame with the bounding boxes and text drawn
        """
        #global output
        output = None

        for qr in self.codes:
            (x, y, w, h) = qr["rect"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, qr["text"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            output = qr["text"]

            if output != None:
                #output = qr["text"]
                return frame, output 

        return frame, output  
        
    def get_qr_text(self): 
        text = self.codes[0]["text"]
        text_array = text.split("\n")
        if len(text_array) != 3: 
            return {
                "success": False,
                "has_questions": False,
                "error_message": "Could not parse newlines correctly",
                "format": "Questions:\n Word word? Word word? Word word?\n Date; Time; device_id; sensor_id; longitude; latitude",
                "text": text
            }
        else: 
            questions = text_array[1]
            d = text_array[2].split("; ") # HERE
            if len(d) != 5: 
                return {
                        "success": False,
                        "has_questions": True,
                        "error_message": "Could not parse semicolons correctly in the third line",
                        "questions": questions,
                        "format": "Date; Time; device_id; sensor_id; longitude; latitude",
                        "text": text
                    }
            return {
                "success": True,
                "questions" : questions,
                "date" : d[0],
                "time" : d[1],
                "device_id" : d[2],
                "sensor_id" : d[3],
                "longitude" : d[4],
                "latitude" : d[5],
                "text": text
            }
            