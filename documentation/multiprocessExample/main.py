import multiprocessing as mp
# import pathos.multiprocessing as pm
import time

import counter
import printer


# If the queue is empty then the exit was not requested
def exit_requested(requestQueue):
    return not requestQueue.empty()


def counter_worker(thousands, pause, exitRequest, pipelineOut):
    """
    This function is used for multiprocessing.
    The controller of the process(es) will be referred to as the manager (just to make it easier).

    Parameters
    ----------
    thousands: Data for class constructor
    pause: A mutex lock used to pause/resume this process
    exitRequest: Queue used for indicating that this process should exit
    pipelineOut: Queue used for producer data output
        If this was also a consumer we would have an input queue

    Returns
    -------

    """
    print("Counter start!")

    # Instantiate a class object for this process
    counterClass = counter.Counter(thousands)

    while (True):

        # If the manager acquires the lock instead,
        # processes will end up blocking here
        # This is a turnstile pattern
        pause.acquire()
        pause.release()

        # Do the work
        # Do not unpack the data here, unpack it in the class method
        # All work should be done in the class method!
        # This wrapper function (counterWorker) should only be doing process things!
        # Single Responsibility Principle
        ret, outData = counterClass.count_up()
        # outData should also be the only thing coming out, already packed and ready to go

        # Something went wrong so we'll skip insertion
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
        # Basically what's happening here is "cache invalidation"
        # i.e. I checked a value and I'm going to make a decision
        # but in the background that value has changed (but I don't know that)
        # Writes and reads are safe because queues have their own internal locks
        if (exit_requested(exitRequest)):
            break

        # Better example of the above (C++ pseudo-code):

        # bool exitRequest; // Global
        # std::lock_guard<std::mutex> exitMtx; // Global
        # ...
        # exitMtx.lock(); // Entering critical section
        # if (exitRequest) {
        #     exitMtx.unlock(); // Leaving critical section
        #     break; // Exiting loop, so we won't accidentally unlock twice
        # }
        # exitMtx.unlock() // Leaving critical section

    # Once the process reaches the end of the function it will die automatically
    print("Counter finished!")
    return


def printer_worker(prefix, pause, exitRequest, pipelineIn, pipelineOut):
    """
    This function is used for multiprocessing.
    The controller of the process(es) will be referred to as the manager (just to make it easier).

    Parameters
    ----------
    prefix: Data for class constructor
    pause: A mutex lock used to pause/resume this process
    exitRequest: Queue used for indicating that this process should exit
    pipelineIn: Queue used for consumer data input
    pipelineOut: Queue used for producer data output

    Returns
    -------

    """
    print("Printer start!")

    # Instantiate a class object for this process
    printerClass = printer.Printer(prefix)

    while (True):

        # Pause turnstile
        pause.acquire()
        pause.release()

        # Queues are process-safe, so no need to protect it
        # IMPORTANT: If the queue is empty it will be stuck here forever,
        # This is bad if an exit is requested, so a way to get it unstuck
        # is to have the manager manually push a few items (see manager below)
        inData = pipelineIn.get()

        # Do the work (in the class, not in this wrapper function)
        # Single Responsibility Principle
        ret, outData = printerClass.print(inData)
        # outData should also be the only thing coming out, already packed and ready to go

        # Something went wrong so we'll skip
        if (not ret):
            continue

        print("Inserting data: " + outData)
        # Gets stuck if full
        pipelineOut.put(outData)

        # Check whether a reset was called
        if (exit_requested(exitRequest)):
            break

    # Once the process reaches the end of the function it will die automatically
    print("Printer finished!")
    return


if __name__ == "__main__":

    # This is what I will call the manager
    # The manager is responsible for any supporting locks and queues
    # and creating and initializing class objects
    # IMPORTANT: Class globals are NOT retained across processes and setting them at runtime is HIGHLY DISCOURAGED
    counterPrinterPause = mp.Lock()
    counterPrinterPipeline = mp.Queue(5)  # Queue of size 5
    printerMainPipeline = mp.Queue()
    counterPrinterStop = mp.Queue(1)  # Queue of size 1

    # For class constructor in each process
    counters = [1, 3, 4, 2]

    numCounterProcesses = 3  # 3 for balanced, 2 for printer waiting for items, 4 for counter waiting to place items

    # For class constructor in each process
    printers = ["Y ", "k "]

    numPrinterProcesses = len(printers)


    # Prepare processes
    c = []
    for i in range(0, numCounterProcesses):
        c.append(mp.Process(target=counter_worker,
                            args=(counters[i], counterPrinterPause, counterPrinterStop, counterPrinterPipeline,)))
    # Pause and stop are shared, so calling a pause/stop will stop the entire group of counters and printers
    # Of course, they can be not shared (simply use two different pauses and two different stops)
    p = []
    for i in range(0, numPrinterProcesses):
        p.append(mp.Process(target=printer_worker,
                            args=(printers[i], counterPrinterPause, counterPrinterStop, counterPrinterPipeline, printerMainPipeline)))


    # Start processes
    for i in range(0, numCounterProcesses):
        c[i].start()
    for i in range(0, numPrinterProcesses):
        p[i].start()


    # Run for some time and then stop
    time.sleep(2)
    counterPrinterPause.acquire()
    print("Paused")
    time.sleep(4)
    print("Resumed")
    counterPrinterPause.release()
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
    counterPrinterStop.put(True)
    # Any item to make the queue non-empty will do,
    # but True is used here to make it clearer


    # In case the processes are stuck on a queue
    # The maximum number of processes stuck on a single in or out queue
    # is the number of running processes (obviously)
    # They may all be waiting for data (stuck on empty queue), so we put in that number of elements
    # They may all be trying to output data (stuck on full queue), so we pop that number of elements
    # We don't really care about the data any more because the whole system is halting
    if (counterPrinterPipeline.full()):
        for i in range(0, numCounterProcesses):
            # There may be the case that there are more producers than the queue max size
            try:
                # This is dangerous because there's a race condition
                # What if the producer takes more than 3 seconds to finish up?
                counterPrinterPipeline.get(timeout=3)
            except mp.Queue.queue.Empty:
                break
    elif (counterPrinterPipeline.empty()):
        for i in range(0, numPrinterProcesses):
            # There may be the case that there are more consumers than the queue max size
            try:
                # Race condition again
                counterPrinterPipeline.put(timeout=3)
            except mp.Queue.queue.Full:
                break


    # Wait for all the processes to finish so it doesn't print over top of each other
    time.sleep(1)

    # Let's check the contents of the queue
    # This also happens to dump the contents of the queue, so we can reuse it later
    print("Queue contents:")
    while (True):
        try:
            print(printerMainPipeline.get_nowait())
        except:
            break

    # We can reset the stop in case we want to reuse it
    # Alternatively create a new stop queue
    counterPrinterStop.get()
    print("Done!")
