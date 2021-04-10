import multiprocessing as mp
import time
import random

# TODO Deprecated

"""
This is an example of a multiprocessed consumer
A producer-consumer will have both an input and output, shouldn't be too hard
"""

class Printer:
    """
    Concatenates a prefix to the input queue and prints it
    Consumer: Gets a value from the provided queue
    """

    # These are all static (global) variables
    # They're all public in this example, but should probably be private
    # The manager class would be a "friend", in C++ parlance

    # Globals, if not initialized directly here, will most likely be initialized in the manager
    prefixStart = ""
    pipelineIn = None
    # This lock is used by the manager to start/stop the class process
    managerTurnstile = mp.Lock()
    # For exiting the class process
    exitRequest = False
    exitLock = mp.Lock()

    def __init__(self, prefixEnd):

        self.prefix = self.prefixStart + prefixEnd


    def run(self):

        p = mp.Process(target=self.printing())
        p.start()


    # The working function
    def printing(self):

        print("Printer start!")
        while (True):

            # If the manager acquires the lock instead,
            # all class processes will end up blocking here
            self.managerTurnstile.acquire()
            self.managerTurnstile.release()

            # This is process-safe, so no need to protect it
            # IMPORTANT: If the queue is empty it will be stuck here forever
            # See (*) below for a solution (in the manager class)
            value = self.pipelineIn.get()

            # Working very hard I see
            time.sleep(0.3)

            stringToPrint = self.prefix + str(value)
            print(stringToPrint)
            # Place into an outgoing pipeline/queue if this was also a producer

            # Check whether a reset was called
            # It is shared, so it is critical and protection is required
            self.exitLock.acquire()
            if (self.exitRequest):
                # Release the lock before breaking out so other class processes can check as well
                self.exitLock.release()
                break
            self.exitLock.release()

        # Once the class process reaches the end of the function it will die automatically
        print("Printer finished!")


class PrinterManager:
    """
    Manages multiple Printer class processes
    While the managers here appear to be very similar,
    there may be a case where there are additional
    different processes (functions) that need to run as well
    Also, class globals are different
    """

    # These should be the only manager static members (global variables)
    # All others are in the class itself
    printerClasses = []
    processCount = 0


    # Class process globals are set here
    def __init__(self, processCount, pipelineIn, prefixStart):

        # Manager class is touching privates, that's okay!
        Printer.pipelineIn = pipelineIn
        Printer.prefixStart = prefixStart
        # Entering critical section
        Printer.exitLock.acquire()
        Printer.exitRequest = False
        # Exiting critical section
        Printer.exitLock.release()

        # Create and start the processes
        self.processCount = processCount
        for i in range(0, self.processCount):
            self.printerClasses.append(Printer(str(random.random()) + " "))


    def start(self):

        for it in self.printerClasses:
            it.run()


    def stop(self):

        # Lock the turnstile
        Printer.managerTurnstile.acquire()
        print("Printers stopped")


    def resume(self):

        # Unlock the turnstile
        Printer.managerTurnstile.release()
        print("Printers resumed")


    # Reset should NEVER be called while the manager is stopped
    def reset(self):

        # Entering critical section
        Printer.exitLock.acquire()
        Printer.exitRequest = True
        # Exiting critical section
        Printer.exitLock.release()
        # Now the class processes will automagically stop on their own

        # Resetting the manager's globals
        self.printerClasses = []
        self.processCount = 0
