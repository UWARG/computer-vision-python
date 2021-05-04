from modules.geolocation.geolocation import Geolocation
import multiprocessing as mp
# For exception handling
import queue


def exit_requested(requestQueue):
    return not requestQueue.empty()


def geolocation_locator_worker(pause, exitRequest, pipelineIn, pipelineOut, pipelineOutLock):

    print("Start Geolocation")
    locator = Geolocation()

    while True:

        pause.acquire()
        pause.release()

        # Pixel coordinates of tents and plane data
        coordinates = pipelineIn.get()
        # Check for valid input
        if (coordinates is None):
            continue

        ret, location = locator.run_locator(coordinates)

        # Something has gone wrong, skip
        if (not ret):
            continue

        # External lock required as geolocation_output_worker will dump the queue (without using the internal one)
        pipelineOutLock.acquire()
        pipelineOut.put(location)
        pipelineOutLock.release()

        if (exit_requested(exitRequest)):
            break


    print("Stop Geolocation")
    return


def geolocation_output_worker(pause, exitRequest, pipelineIn, pipelineOut, pipelineInLock):

    print("Start Geolocation Output")
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


    print("Stop Geolocation Output")
    return
