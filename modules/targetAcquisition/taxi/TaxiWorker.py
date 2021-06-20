from modules.targetAcquisition.taxi import Taxi

def taxi_worker(pause, exitRequest, pipelineIn, pipelineOut):
    print("Start Taxi")
    taxi = Taxi()

    while True:
        pause.acquire()
        pause.release()
        
        frame = pipelineIn.get()
        taxi.main(frame)

        pipelineOut.put()

        if not exitRequest.empty():
            return


