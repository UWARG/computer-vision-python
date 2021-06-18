from modules.commandModule.commandModule import CommandModule
from modules.commandModule.commandFns import write_pigo, read_pogi

POGI_DIR = ""
PIGO_DIR = ""

def pogi_subworker(pipelineOut, POGI_DIR):

    # get the pogi data
    changed, pogiData = read_pogi(POGI_DIR)

    # if pogi has changed, output to the pipeline
    if changed:
        pipelineOut.put(pogiData)


def flight_command_worker(pipelineIn, pipelineOut, pause, exitRequest):
    
    """
    Worker function for PIGO and POGI data transactions

    Parameters
    ----------
    pipelineIn : multiprocessing.Queue
        input pipeline for pigoData
    pipelineOut : multiprocessing.Queue
        output pipeline for pogiData
    pause : multiprocessing.Lock
        a lock shared between worker processes
    exitRequest : multiprocessing.Queue
        a queue that determines when the worker process stops
    """

    while True:

        pause.acquire()
        pause.release()

        # POGI Logic
        pogi_subworker(pipelineOut, POGI_DIR)

        if not exitRequest.empty():
            return
        
        # PIGO Logic
        # pipelineIn gives [[x,y], [range]]
        if pipelineIn.empty:
            continue
        
        data = pipelineIn.get()
        # Cache data here
        newPigo = {
            'gpsCoordinates': data[0]
        }

        write_pigo(newPigo)
