import multiprocessing as mp
# import pathos.multiprocessing as pm
import time
import counter


# If the queue is empty then the exit was not requested
def exitRequested(requestQueue):
    return not requestQueue.empty()

def counterWorker(counterClass, pause, exitRequest, pipelineOut):
    """
    This function is used for multiprocessing.
    The controller of the process(es) will be referred to as the manager (just to make it easier).

    Parameters
    ----------
    counterClass: The class object
    pause: A mutex lock used to pause/resume this process
    exitRequest: Queue used for indicating that this process should exit
    pipelineOut: Queue used for producer data output
        If this was also a consumer we would have an input queue

    Returns
    -------

    """
    print("Counter start!")
    while (True):

        # If the manager acquires the lock instead,
        # processes will end up blocking here
        # This is a turnstile pattern
        pause.acquire()
        pause.release()

        # If the class were also a consumer, this code would exist
        # Queues are process-safe, so no need to protect it
        # IMPORTANT: If the queue is empty it will be stuck here forever,
        # This is bad if an exit is requested, so a way to get it unstuck
        # is to have the manager manually push a few items (see manager below)

        # inData = pipelineIn.get()

        # Do the work
        # Do not unpack the data here, unpack it in the class method
        # All work should be done in the class method!
        # This wrapper function should only be doing process things!
        # Single Responsibility Principle
        ret, outData = counterClass.countUp()  # .countUp(inData) if it's also a consumer

        # Something went wrong so we'll skip
        if (not ret):
            continue

        print("Inserting data: " + str(outData))
        # Queues are process-safe, so no need to protect it
        # IMPORTANT: If the queue is full it will be stuck here forever
        # This is bad if an exit is requested, so a way to get it unstuck
        # is to have the manager manually pop a few items (see manager below)
        pipelineOut.put(outData)

        # Check whether a reset was called
        # It is shared, which means it's a critical section,
        # and so this is technically bad because it's not protected by a lock
        # HOWEVER: Sharing memory between processes is a pain,
        # and the worst case is that we end up
        # processing an extra item before exiting and dying
        if (exitRequested(exitRequest)):
            break

        # Better example of the above (C++ pseudo-code):

        # bool exitRequest; // Global
        # mutexlock exitMtx; // Global
        # ...
        # exitMtx.lock(); // Entering critical section
        # if (exitRequest) {
        #     exitMtx.unlock(); // Leaving critical section
        #     break; // Exiting loop, so we won't accidentally unlock twice
        # }
        # exitMtx.unlock() // Leaving critical section

    # Once the process reaches the end of the function it will die automatically
    print("Counter finished!")


if __name__ == "__main__":

    # This is what I will call the manager
    # The manager is responsible for any supporting locks and queues
    # and creating and initializing class objects
    # IMPORTANT: Class globals are NOT retained across processes and setting them at runtime is HIGHLY DISCOURAGED
    counters = [
        counter.Counter(1),
        counter.Counter(4),
        counter.Counter(3)
    ]
    counterPause = mp.Lock()
    counterToMainPipeline = mp.Queue()
    counterStop = mp.Queue(1)  # Queue of size 1

    numProcesses = 3

    # Prepare processes
    p = []
    for i in range(0, numProcesses):
        p.append(mp.Process(target=counterWorker,
                            args=(counters[i], counterPause, counterStop, counterToMainPipeline,)))

    # Start processes
    for i in range(0, numProcesses):
        p[i].start()

    # Run for some time and then stop
    time.sleep(2)
    counterPause.acquire()
    print("Paused")
    time.sleep(4)
    print("Resumed")
    counterPause.release()
    time.sleep(2)

    # Stop must be requested while the processes are running!
    # "Well, why not release the pause lock then?"
    # What if it isn't paused? Double release will cause an exception
    # "Okay, well if it isn't paused that's okay, just catch the exception"
    # This adds complexity to both the manager and the process function, which is bad
    # "Not as bad as crash, also it will catch double resumes"
    # Silent error is pretty bad, especially if the manager acquires a pause lock while
    # a process is still inside the (empty) critical section. Can you see how this would happen?
    #  - Xierumeng's brain
    counterStop.put(True)  # Any item will do, but True is used here to make it clearer

    # In case the processes are stuck on a queue
    # The maximum number of processes stuck on a single in or out queue
    # is the number of running processes (obviously)
    # They may all be waiting for data (stuck on empty queue), so we put in that number of elements
    # They may all be trying to output data (stuck on full queue), so we pop that number of elements
    for i in range(0, numProcesses):

        # pipelineIn.put(False)
        counterToMainPipeline.get()

    # Wait for all the processes to finish so it doesn't print over top of each other
    time.sleep(1)

    # Let's check the contents of the queue
    print("Queue contents:")
    while (True):
        try:
            print(counterToMainPipeline.get_nowait())
        except:
            break

    print("Done!")

    # We can reset the stop in case we want to reuse it
    # Alternatively create a new stop queue
    counterStop.get()
