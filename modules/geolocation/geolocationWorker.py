from modules.geolocation.geolocation import Geolocation
import multiprocessing as mp
import queue  # For exception handling
import time  # For sleep()
import logging


def exit_requested(requestQueue):
    return not requestQueue.empty()


def geolocation_locator_worker(pause, exitRequest, pipelineIn, pipelineOut, pipelineOutLock):
    
    logger = logging.getLogger()
    logger.debug("geolocation_locator_worker: Start Geolocation Locator")
    
    locator = Geolocation()
    # Competition
    locator.set_constants()

    while True:

        pause.acquire()
        pause.release()

        # Merged Data
        merged_data = pipelineIn.get()
        # Check for valid input
        if (merged_data is None):
            continue

        ret, location = locator.run_locator(merged_data.telemetry, merged_data.image)

        # Something has gone wrong, skip
        if (not ret):
            continue

        # External lock required as geolocation_output_worker will dump the queue (without using the internal one)
        # Keep attempting if the pipeline is full
        while True:

            try:

                pipelineOutLock.acquire()
                pipelineOut.put_nowait(location)
                pipelineOutLock.release()
                break

            except queue.Full:

                pipelineOutLock.release()
                time.sleep(0.5)  # Hopefully allow Geolocation Output to empty the queue a bit


        if (exit_requested(exitRequest)):
            break


    logger.debug("geolocation_locator_worker: Stop Geolocation Locator")
    return


def geolocation_output_worker(pause, exitRequest, pipelineIn, pipelineOut, pipelineInLock):

    logger = logging.getLogger()
    logger.debug("geolocation_output_worker: Start Geolocation Output")

    locator = Geolocation()

    while True:

        pause.acquire()
        pause.release()

        # Dump the entirety of the input queue
        locations = []
        pipelineInLock.acquire()
        while True:

            try:

                # This does not use the internal lock, which is why an external one is required
                # get_nowait() used to prevent blocking on empty queue
                location = pipelineIn.get_nowait()
                # Check for valid input
                if (location is None):
                    continue

                locations.append(location)

            # Queue is empty, done
            except queue.Empty:

                break

        pipelineInLock.release()

        ret, bestOutput = locator.run_output(locations)

        # Something has gone wrong, skip
        if (not ret):
            continue

        pipelineOut.put(bestOutput)

        if (exit_requested(exitRequest)):
            break

    # write to file here? using locations list [], does locations need to be outside of the loop? global
    # call function using locations here
    locator.write_locations(locations)

    logger.debug("geolocation_output_worker: Stop Geolocation Output")
    return
