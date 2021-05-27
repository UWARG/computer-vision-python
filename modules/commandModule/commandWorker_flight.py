from modules.commandModule.commandModule import CommandModule
from modules.commandModule.commandFns import write_pigo, read_pogi

POGI_DIR = ""
PIGO_DIR = ""

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

        # get the pogi data
        pogiData = read_pogi()

        # output to pipeline
        # note: no null checking required since CommandModule returns None for fields that are unreadable/inaccesible
        #       if all fields are None, then pogiData will be a dict with all elements as {'example_key' : None}
        pipelineOut.put(pogiData)

        if not exitRequest.empty():
            return