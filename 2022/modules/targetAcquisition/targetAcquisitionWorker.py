from modules.targetAcquisition.targetAcquisition import TargetAcquisition
import logging

def targetAcquisitionWorker(pause, exitRequest, mergedDataPipelineIn, coordinatesTelemetryPipelineOut):
    logger = logging.getLogger()
    logger.debug("targetAcquisitionWorker: Start Target Acquisition Module")
    
    targetAcquisition = TargetAcquisition()
    
    while True:
        if not exitRequest.empty():
            break
        
        pause.acquire()
        pause.release()

        curr_frame = mergedDataPipelineIn.get()

        if curr_frame is None:
            continue
        
        # Set the current frame
        targetAcquisition.set_curr_frame(curr_frame)

        # Run model
        res, coordinatesAndTelemetry = targetAcquisition.get_coordinates()
        if not res:
            continue
            
        coordinatesTelemetryPipelineOut.put(coordinatesAndTelemetry)
        
        logger.info("targetAcquisitionWorker: Found a pylon: " + str(coordinatesAndTelemetry))

    logger.debug("targetAcquisitionWorker: Stop Target Acquisition Module")

