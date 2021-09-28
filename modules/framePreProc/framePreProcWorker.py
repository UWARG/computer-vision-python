from framePreProc import FramePreProc

lastTelemetry = None

def worker(telemetryDataObject, pipelineOut):

    """

    Takes Euler data and passes it to the pipeline if the data passes through the framePreProc filter

    Parameters
    ----------

    telemetryDataObject: MergedData object
        Includes the telemetry data + euler angles data via .telemetry

    pipelineOut: Queue?
        Destination of telemetry data successfully passing through the filter

    """

    global lastTelemetry

    eulerAnglesDict = telemetryDataObject.telemetry['eulerAnglesOfCamera']

    if lastTelemetry:
        lastEulerAnglesDict = lastTelemetry['eulerAnglesOfCamera']
        framePreProcObject = FramePreProc(eulerAnglesDict,lastEulerAnglesDict)
        if framePreProcObject.filter():
            lastTelemetry = telemetryDataObject.telemetry
            pipelineOut.put(telemetryDataObject)

    else:
        lastTelemetry = telemetryDataObject.telemetry

