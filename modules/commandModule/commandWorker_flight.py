from modules.commandModule.commandModule import CommandModule

PIGO_DIR = ""

def flight_command_worker(pipelineIn, pipelineOut):
	command = CommandModule(pigoFileDirectory=PIGO_DIR)
	# pipelineIn gives [[x,y], [range]]
	if pipelineIn.empty:
	   continue
	
	data = pipelineIn.get()
	# Cache data here
	gps_coordinates = data[0]

	command.set_gps_coordinates(gps_coordinates)

	#TODO: Write logic to get POGI

	
