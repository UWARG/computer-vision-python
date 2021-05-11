from modules.commandModule.commandFns import write_pigo

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

	