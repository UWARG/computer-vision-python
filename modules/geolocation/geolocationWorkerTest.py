import time
import multiprocessing as mp
import random

import geolocationWorker
from modules.mergeImageWithTelemetry.mergedData import MergedData

"""
This test is mainly benchmarking multiprocessing performance for résumé purposes.
"""
if __name__ == "__main__":

    maxWorkers = 4
    for k in range(1, maxWorkers + 1):
        print("Start run " + str(k))

        # Setup data
        print("Setting up data")
        pause = mp.Lock()
        exitRequest = mp.Queue()
        inputPipeline = mp.Queue()
        outputPipeline = mp.Queue()
        outputLock = mp.Lock()

        # Initial input data
        dataCount = 4000 * k
        for i in range(0, dataCount):
            camera_euler = {
                'yaw': 0,
                'pitch': 90,
                'roll': 0
            }
            plane_euler = {
                'yaw': random.randint(0, 359),
                'pitch': random.randint(-15, 15),
                'roll': random.randint(-45, 45)
            }
            gps = {
                "longitude": random.randint(-100, 100),
                "latitude": random.randint(-100, 100),
                "altitude": random.randint(10, 100)
            }
            telemetry = {
                "eulerAnglesOfPlane": plane_euler,
                "eulerAnglesOfCamera": camera_euler,
                "gpsCoordinates": gps
            }

            coordinates = []
            for j in range(0, 5):
                x = random.randint(0, 1000)
                y = random.randint(0, 1000)
                coordinates.append([x, y])

            inputPipeline.put(MergedData(coordinates, telemetry))

        # Setup workers
        processCount = 1 * k  # Number of processes
        p = []
        for i in range(0, processCount):
            p.append(mp.Process(target=geolocationWorker.geolocation_locator_worker,
                     args=(pause, exitRequest, inputPipeline, outputPipeline, outputLock)))

        # Start processes
        print("Starting workers")
        for i in range(0, processCount):
            p[i].start()

        # Adjust as required, generally we want to stop before emptying the input pipeline
        workingTime = 10
        time.sleep(workingTime)

        # Stop processes
        exitRequest.put(True)

        # Dump the queue
        """
        print("Dumping queue")
        j = 0
        while (True):
            try:
                # A guaranteed delay is necessary, otherwise this loop will collide with itself!
                # I'll investigate how short this can be.
                time.sleep(0.1)
                data = outputPipeline.get_nowait()
                # print(data)
                j += 1
            except:
                break
        """

        # Results
        print(str(k) + " workers completed " + str(outputPipeline.qsize()) + " in " + str(workingTime) + " seconds")

    print("Done!")
