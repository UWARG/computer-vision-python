from modules.commandModule.commandModule import CommandModule
import logging
from directories import PIGO_DIR, POGI_DIR


def taxi_command_worker_first(turn_command_data):
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
    logger.debug("commandWorker_taxi_first/taxi_command_worker_first: Started")

    command = CommandModule(pigoFileDirectory=PIGO_DIR, pogiFileDirectory=POGI_DIR)
    command.set_ground_commands(turn_command_data)

    logger.debug("commandWorker_taxi_first/taxi_command_worker_first: Finished")

def command_taxi_worker_continuous(pause, exitRequest, pipelineIn):
    logger = logging.getLogger()
    logger.debug("commandWorker_taxi_first/command_taxi_worker_continuous: Started")

    command = CommandModule(pigoFileDirectory=PIGO_DIR, pogiFileDirectory=POGI_DIR)
    
    while True:
        pause.acquire()
        pause.lock()

        # .get() waits until something is available from the pipeline,
        # so no need to continuously loop around the while True waiting for data
        taxiCommands = pipelineIn.get()

        # Here's a check just in case however
        if taxiCommands is None:
            continue

        command.set_ground_commands(taxiCommands)

        if not exitRequest.empty():
            break
    
    logger.debug("commandWorker_taxi_first/command_taxi_worker_continuous: Finished")