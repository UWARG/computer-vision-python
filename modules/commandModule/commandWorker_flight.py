from modules.commandModule.commandModule import CommandModule

PIGO_DIR = ""

def flight_command_worker(pipelineIn, pipelineOut):
	command = CommandModule(pigoFileDirectory=PIGO_DIR)
	# pipelineIn gives [[x,y], [range]]
	if pipelineIn.empty:
	   continue
	
	data = pipelineIn.get()
	# Cache data here
	ground_command = {
		'gpsCoordinates': [data[0], data[1]]
	}

	command.set_gps_coordinates(ground_command)

	#TODO: Write logic to get POGI

	
