import json

from modules.commandModule.commandModule import CommandModule

def flight_command_worker(pipelineIn, pipelineOut, pigo_dir=""):
	command = CommandModule(pigoFileDirectory=pigo_dir)
	while True:
		# pipelineIn gives [[x,y], [range]]
		if pipelineIn.empty:
		   continue

		data = pipelineIn.get()
		gps_coordinates = data[0]

		# Cache gps_coordinates into a JSON file
		with open("temp_pylon_gps", "w") as pylon_gps_file:
			json.dump(gps_coordinates, pylon_gps_file)
		
		command.set_gps_coordinates(gps_coordinates)
		return

	#TODO: Write logic to get POGI

	
