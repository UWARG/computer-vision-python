from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
import cv2

import logging

def targetAcquisitionCompWorker(pause, exitRequest, mergedDataPipelineIn):
    logger = logging.getLogger()
    logger.debug("targetAcquisitionWorker: Start Target Acquisition Module")

    targetAcquisition = TargetAcquisition()

    while True:
        if not exitRequest.empty():
            break

        pause.acquire()
        pause.release()

        image = mergedDataPipelineIn.get()

        if image is None:
            continue

        # Set the current frame
        targetAcquisition.set_curr_frame(MergedData(image, dict()))

        # Run model
        boxes = targetAcquisition.get_coordinates()
        for box in boxes:
            image = cv2.rectangle(image, box[0], box[1], (0, 255, 255), 2)

        cv2.imshow("Pew pew box", image)

        # OpenCV doesn't allow you to access a camera without a camera release, So feel free to replace this
        # bottom with however the video stream will quit (right now it quits on spacebar)
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.stop()
            break

        logger.info("targetAcquisitionCompWorker: Found a person")

    logger.debug("targetAcquisitionWorker: Stop Target Acquisition Module")

