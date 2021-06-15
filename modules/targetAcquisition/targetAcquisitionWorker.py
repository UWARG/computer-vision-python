from modules.targetAcquisition.targetAcquisition import TargetAcquisition


def targetAcquisitionWorker(pause, exitRequest, pipelineIn, pipelineOut):
    print("start target acquisition")
    targetAcquisition = TargetAcquisition()
    while True:
        if not exitRequest.empty():
            return
        
        pause.acquire()
        pause.release()

        curr_frame = pipelineIn.get()

        if curr_frame is None:
            continue

        coordinates = targetAcquisition.get_coordinates(curr_frame)
        if coordinates is None:
            continue

        pipelineOut.put(coordinates)


