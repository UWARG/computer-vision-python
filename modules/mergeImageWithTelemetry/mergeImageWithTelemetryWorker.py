from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry

def pipelineMergeWorker(pause, exitRequest, imagePipelineIn, telemetryPipelineIn, pipelineOut):
    print("start pipeline merge")
    mergeImageWithTelemetry = MergeImageWithTelemetry()
    shouldGetImage = True
    curImage = None

    while True: 
        if not exitRequest.empty():
            print("end pipeline merge")
            return
        
        pause.acquire()
        pause.release()
        
        if shouldGetImage:
            try: 
                curImage = imagePipelineIn.get_noawait()
            except: 
                curImage = None

        if curImage is None:
            continue

        # put current elements of the telemetry pipeline into the queue
        telemetryPipelineIn.acquire()
        while True: 
            try: 
                telemetryData = telemetryPipelineIn.get_noawait()
                mergeImageWithTelemetry.put_back(telemetryData)
            except: 
                break
        telemetryPipelineIn.release()

        [success, mergedData] = mergeImageWithTelemetry.merge_with_closest_telemetry(curImage.timestamp, curImage.data)

        if success: 
            pipelineOut.put(mergedData)
            shouldGetImage = True # curImage has been matched with a telemetry
        else: 
            shouldGetImage = False # curImage has not been matched with a telemetry and we should keep it as the current image


        

