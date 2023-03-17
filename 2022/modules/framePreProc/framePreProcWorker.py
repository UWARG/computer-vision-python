from framePreProc import FramePreProc

lastTelemetry = None

def worker(pipelineIn, pipelineOut):

    """

    Takes Euler data and passes it to the pipeline if the data passes through the framePreProc filter

    Parameters
    ----------

    pipelineIn: Queue holding MergedData object
        Includes the telemetry data + euler angles data via .telemetry

    pipelineOut: Queue
        Destination of telemetry data successfully passing through the filter

    """

    frame_proc  = FramePreProc()
    while True:

        dataObj = pipelineIn.get()
        eulerAnglesDict = dataObj.telemetry['eulerAnglesOfCamera']

        filter_pass = frame_proc.filter(eulerAnglesDict)
        frame_proc.update_last_dict(eulerAnglesDict)

        if filter_pass:
            pipelineOut.put(dataObj)

        else:
            continue

