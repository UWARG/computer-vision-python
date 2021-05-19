from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry

def pipelineMergeWorker(pause, exitRequest, imagePipelineIn, telemetryPipelineIn, pipelineOut):
    print("start pipeline merge")
    mergeImageWithTelemetry = MergeImageWithTelemetry()
    shouldGetImage = True
    curImage = None

    while True: 
        if not exitRequest.empty():
            return
        
        pause.acquire()
        pause.release()
        
        if shouldGetImage:
            curImage = imagePipelineIn.get()

        if curImage is None:
            continue

        # put current elements of the telemetry pipeline into the queue
        while not telemetryPipelineIn.empty(): 
        
            if not exitRequest.empty(): 
                return 
            
            pause.acquire()
            pause.release()

            telemetryData = telemetryPipelineIn.get()

            mergeImageWithTelemetry.put_back(telemetryData)

        [success, mergedData] = mergeImageWithTelemetry.merge_with_closest_telemetry(curImage.timestamp, curImage.data)

        if success: 
            pipelineOut.put(mergedData)
            shouldGetImage = True # curImage has been matched with a telemetry
        else: 
            shouldGetImage = False # curImage has not been matched with a telemetry and we should keep it as the current image


        

