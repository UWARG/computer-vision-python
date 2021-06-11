#from modules.targetAcquisition.taxi import Taxi
#from Taxi import Taxi
from Taxi import Taxi

def taxi_worker(pause, exitRequest, pipelineIn, pipelineOut):
    print("Start Taxi")
    taxi = Taxi()

    while True:
        pause.acquire()
        pause.release()
        
        frame = pipelineIn.get()
        if taxi.main(frame).latestDistance == 0:
            pipelineOut.put(taxi.main(frame))

        if not exitRequest.empty():
            return


