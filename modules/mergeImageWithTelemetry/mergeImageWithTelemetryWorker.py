from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry

def pipelineMergeWorker(pause, exitRequest, imagePipelineIn, telemetryPipelineIn, pipelineOut):
    print("start pipeline merge")
    mergeImageWithTelemetry = MergeImageWithTelemetry()
    curImage = None

    while True: 
        if not exitRequest.empty():
            print("end pipeline merge")
            return
        
        pause.acquire()
        pause.release()

        # put current elements of the telemetry pipeline into the queue
        telemetryPipelineIn.acquire()
        while True: 
            try: 
                telemetryData = telemetryPipelineIn.get_noawait()
                mergeImageWithTelemetry.put_back_telemetry(telemetryData)
            except: 
                break
        telemetryPipelineIn.release()
        
        if mergeImageWithTelemetry.should_get_image():
            try: 
                curImage = imagePipelineIn.get_noawait()
                mergeImageWithTelemetry.set_image(curImage)
            except: 
                curImage = None

        if curImage == None:
            continue

        [success, mergedData] = mergeImageWithTelemetry.get_closest_telemetry(curImage.timestamp, curImage.data)

        if success: 
            pipelineOut.put(mergedData)
