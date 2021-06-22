from Taxi import Taxi

def taxi_worker(pause, exitRequest, pipelineIn, pipelineOut):
    """
        Taxi Worker Process 
        Implements multiprocessing for the taxi module
        Gets frame from the input pipeline, performs target acquisition and puts a stop or move command into the output pipeline
        
        Parameters
        ----------
        pause : multiprocessing.Lock
          a lock shared between worker processes
        exitRequest : multiprocessing.Queue
          a queue that determines when the worker process stops
        pipelineIn : multiprocessing.Queue
          input pipeline
        pipelineOut : multiprocessing.Queue
          output pipeline
        
        Returns
        -------
        None
        """
    
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


