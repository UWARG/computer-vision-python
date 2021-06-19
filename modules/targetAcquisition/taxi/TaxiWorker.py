from Taxi import Taxi

def taxi_worker(pause, exitRequest, pipelineIn, pipelineOut):
    print("Start Taxi")
    taxi = Taxi()

    while True:
        pause.acquire()
        pause.release()
        frame = pipelineIn.get()
        if (not frame == None):
            command = taxi.main(frame)
            if (hasattr(command, latestDistance)) and (command.latestDistance == 0):
                pipelineOut.put(command)

        if not exitRequest.empty():
            return


