import cv2
import numpy as np
from pyzbar.pyzbar import decode

def scan(img):
    gray_img = cv2.cvtColor(image, 0)
    barcode = decode(gray_img)

    # Assume there is only one object in each frame, so return the first decoded message
    for obj in barcode:
        points = obj.polygon
        (x,y,w,h) = obj.rect
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 3)

        barcodeData = obj.data.decode("utf-8")
        barcodeType = obj.type
        return f"Barcode: {barcodeData} | Type: {barcodeType}"