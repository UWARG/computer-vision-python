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
        self.codes = []

    def main(self, frame):
        """
        Runs get_qr_codes to populate self.codes & draw_qr_codes to draw bounding boxes

        Parameters
        ----------
        frame : np.ndarray
            Image to be processed
        """
        self.codes = []
        self.get_qr_codes(frame)
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
        for qr in self.codes:
            (x, y, w, h) = qr["rect"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, qr["text"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame
