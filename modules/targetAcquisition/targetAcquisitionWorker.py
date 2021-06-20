from modules.targetAcquisition.targetAcquisition import TargetAcquisition
import logging

def targetAcquisitionWorker(pause, exitRequest, pipelineIn, pipelineOut):

    logger = logging.getLogger()
    logger.debug("targetAcquisitionWorker: Start Target Acquisition Module")
    
    targetAcquisition = TargetAcquisition()
    
    while True:
        if not exitRequest.empty():
            break
        
        pause.acquire()
        pause.release()

        curr_frame = pipelineIn.get()

        if curr_frame is None:
            continue

        bbox = targetAcquisition.get_coordinates(curr_frame)
        if bbox is None:
            continue
        
        logger.info("targetAcquisitionWorker: Found a box: " + str(coordinates))
        pipelineOut.put(coordinates)

    logger.debug("targetAcquisitionWorker: Stop Target Acquisition Module")

