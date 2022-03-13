from modules.searchExplosive.searchExplosive import SearchExplosive
import logging
import cv2


def searchExplosiveWorker(pause, exitRequest, pipelineIn):
    logger = logging.getLogger()
    logger.debug("searchExplosiveWorker: Start Search Explosive Module")

    while True:
        if not exitRequest.empty():
            break

        pause.acquire()
        pause.release()

        frame = pipelineIn.get()

        if frame is None:
            continue

        # Run the edge detector
        detector = SearchExplosive(frame)
        detector.edge_detection()
        detector.contour_detection()

        if detector.count == 0:
            logger.debug("searchExplosiveWorker: No detected objects")
        cv2.imshow("Video feed", detector.detectedContours)
        cv2.waitKey(1)

    logger.debug("searchExplosiveWorker: Stop Search Explosive Module")
