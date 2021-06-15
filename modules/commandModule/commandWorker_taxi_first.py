from modules.commandModule.commandModule import CommandModule

def taxi_command_worker_first(pipelineIn, pipelineOut):
	"""
	Worker process that gets a turn command [dict] from the search worker and updates the pigo by calling set_ground_command()
	
	Parameters
	----------
	pipelineIn : multiprocessing.Queue
	input pipeline that has a turn command dictionary with format {"heading": bearing, "latestDistance": 0}
	pipelineOut : multiprocessing.Queue
	output pipeline
    
	Returns
	-------
	None
	"""
	
	if pipelineIn.empty:
		print("No data in taxi_command_worker_first pipelineIn")
		return
	
	turn_command_data = pipelineIn.get()
	set_ground_commands(turn_command_data)
