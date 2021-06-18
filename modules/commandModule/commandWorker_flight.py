import json

from modules.commandModule.commandModule import CommandModule
import logging

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

		data = pipelineIn.get()
		gps_coordinates = data[0]

		# Cache gps_coordinates into a JSON file
		with open("temp_pylon_gps", "w") as pylon_gps_file:
			json.dump(gps_coordinates, pylon_gps_file)
		
		command.set_gps_coordinates(gps_coordinates)
		return

		#TODO: Write logic to get POGI
		pipelineOut.put()
	
	logger.debug("flight_command_worker: Stopped Flight Command Module")
