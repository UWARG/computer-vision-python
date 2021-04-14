from modules.targetAcquisition.targetAcquisition import TargetAcquisition


def targetAcquisitionWorker(pause, exitRequest, pipelineIn, pipelineOut):
    print("start target acquisition")
    targetAcquisition = TargetAcquisition()
    while True:
        pause.acquire()
        pause.release()

        curr_frame = pipelineIn.get()
        coordinates = targetAcquisition.get_coordinates(curr_frame)
        pipelineOut.put(coordinates)

        if not exitRequest.empty():
            return
