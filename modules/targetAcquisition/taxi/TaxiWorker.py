from modules.targetAcquisition.taxi import Taxi

def taxi_worker(pause, exitRequest, pipelineIn, pipelineOut):
    print("Start Taxi")
    taxi = Taxi()

    while True:
        pause.acquire()
        pause.release()
        
        values = pipelineIn.get()

        pipelineOut.put()

        if not exitRequest.empty():
            return


