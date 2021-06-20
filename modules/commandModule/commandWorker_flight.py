from modules.commandModule.commandModule import CommandModule
from modules.commandModule.commandFns import write_pigo, read_pogi
import logging

POGI_DIR = ""
PIGO_DIR = ""

def pogi_subworker(pipelineOut, POGI_DIR):

    # get the pogi data
    changed, pogiData = read_pogi(POGI_DIR)

    # if pogi has changed, output to the pipeline
    if changed:
        pipelineOut.put(pogiData)

def flight_command_worker(pause, exitRequest, pipelineIn, pipelineOut, pigo_dir=""):
	
	logger = logging.getLogger()
	logger.debug("flight_command_worker: Started Flight Command Module")

	command = CommandModule(pigoFileDirectory=pigo_dir)
	while True:
        # Kill process if exit is requested
		if not exitRequest.empty():
			break
		
		pause.acquire()
		pause.release()
		
		# pipelineIn gives [[x,y], [range]]
		if pipelineIn.empty():
			continue

    pause.acquire()
    pause.release()

    # POGI Logic
    pogi_subworker(pipelineOut, POGI_DIR)

    # PIGO Logic

    data = pipelineIn.get()
    gps_coordinates = data[0]
    # Cache data here
    command.set_gps_coordinates(gps_coordinates)
	
	logger.debug("flight_command_worker: Stopped Flight Command Module")

