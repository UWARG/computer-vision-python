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


def flight_command_worker(pipelineIn, pipelineOut):
	# pipelineIn gives [[x,y], [range]]
	if pipelineIn.empty:
	   continue
	
	data = pipelineIn.get()
	# Cache data here
	newPigo = {
		'gpsCoordinates': [x, y]
	}

	write_pigo(newPigo)

	#TODO: Write logic to get POGI


def pogi_command_worker(pause, exitRequest, pipelineOut):
    """
    Producer worker function to return data currently in the POGI file

    Parameters
    ----------
    pause : multiprocessing.Lock
        a lock shared between worker processes
    exitRequest : multiprocessing.Queue
        a queue that determines when the worker process stops
    pipelineOut : multiprocessing.Queue
        output pipeline for pogiData
    """

    print("Reading POGI data")

    while True:
        pause.acquire()
        pause.release()

        pogi_subworker(pipelineOut, POGI_DIR)

        if not exitRequest.empty():
            return