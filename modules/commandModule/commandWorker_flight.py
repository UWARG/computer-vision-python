from modules.commandModule.commandModule import CommandModule

def flight_command_worker(pipelineIn, pipelineOut, pigo_dir=""):
	command = CommandModule(pogiFileDirectory="", pigoFileDirectory=pigo_dir)
	while:
		# pipelineIn gives [[x,y], [range]]
		if pipelineIn.empty:
		   continue

		data = pipelineIn.get()
		gps_coordinates = data[0]
		# Cache gps_coordinates here
		
		command.set_gps_coordinates(gps_coordinates)

	#TODO: Write logic to get POGI

	
