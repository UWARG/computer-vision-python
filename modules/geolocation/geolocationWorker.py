from modules.geolocation.geolocation import Geolocation
import multiprocessing as mp
# For exception handling
import queue


def exit_requested(requestQueue):
    return not requestQueue.empty()


def geolocation_worker(pause, exitRequest, pipelineIn, pipelineOut, pipelineOutLock):

    print("Start Geolocation")
    locator = Geolocation()

    while True:

        pause.acquire()
        pause.release()

        # Pixel coordinates of tents and plane data
        coordinates = pipelineIn.get()
        # TODO verify this
        if (not coordinates):
            continue

        ret, location = locator.run_locator(coordinates)

        # Something has gone wrong, skip
        if (not ret):
            continue

        # External lock required as geolocationOutput will dump the queue
        pipelineOutLock.acquire()
        pipelineOut.put(location)
        pipelineOutLock.release()

        if (exit_requested(exitRequest)):
            break


    print("Stop Geolocation")
    return


def geolocation_output(pause, exitRequest, pipelineIn, pipelineOut, pipelineInLock):

    print("Start Geolocation Output")
    locator = Geolocation()

    while True:

        pause.acquire()
        pause.release()

        # Dump the contents of the input queue
        locations = []
        pipelineInLock.acquire()
        while True:

            try:

                location = pipelineIn.get_nowait()
                # TODO verify this
                if (not location):
                    continue

                locations.append(location)

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
