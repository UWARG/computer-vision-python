from modules.commandModule.commandModule import CommandModule

def flight_command_worker(pause, exitRequest, pipelineIn, pipelineOut, pigo_dir=""):
	command = CommandModule(pigoFileDirectory=pigo_dir)
	while True:
        # Kill process if exit is requested
		if not exitRequest.empty():
			return
		
		pause.acquire()
		pause.release()
		
		# pipelineIn gives [[x,y], [range]]
		if pipelineIn.empty():
			continue

		data = pipelineIn.get()
		gps_coordinates = data[0]

		# Cache gps_coordinates here
		
		command.set_gps_coordinates(gps_coordinates)

		#TODO: Write logic to get POGI
		pipelineOut.put()
	
