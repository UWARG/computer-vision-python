from pyzbar import pyzbar
import cv2

class QRScanner:
    def __init__(self):
        self.codes = []

    def main(self, frame):
        self.codes = []
        self.get_qr_codes(frame)
        return self.draw_qr_codes(frame)

    def get_qr_codes(self, frame):
        qrCodes = pyzbar.decode(frame)
        
        for qr in qrCodes:
            (x, y, w, h) = qr.rect
            
            text = '{}: {}'.format(qr.type, qr.data.decode("utf-8"))
            self.codes.append({"text": text, "rect": (x, y, w, h)})
        return frame

    def draw_qr_codes(self, frame):
        for qr in self.codes:
            (x, y, w, h) = qr["rect"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, qr["text"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame
