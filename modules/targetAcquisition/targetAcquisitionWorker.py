from modules.targetAcquisition.targetAcquisition import TargetAcquisition


def targetAcquisitionWorker(pause, exitRequest, mergedDataPipelineIn, coordinatesTelemetryPipelineOut):
    print("start target acquisition")
    targetAcquisition = TargetAcquisition()
    while True:
        if not exitRequest.empty():
            return
        
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


