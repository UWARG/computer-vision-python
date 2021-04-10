import multiprocessing as mp
import time

"""
This is an example of a multiprocessed producer
A producer-consumer will have both an input and output, shouldn't be too hard
"""

class Counter:
    """
    Counts from 0 to a provided maximum and uses a thousands digit
    Producer: Places current count in the provided queue
    """

    # These are all static (global) variables
    # They're all public in this example, but should probably be private
    # The manager class would be a "friend", in C++ parlance

    # Globals, if not initialized directly here, must be initialized before the process is started
    maxCount = 999

    # Class must be constructed before the process is started
    def __init__(self, thousands):

        self.numberToAdd = thousands * 1000
        self.currentCount = self.numberToAdd


    # The working function
    # If this was also a consumer then the function signature would include data input
    def countUp(self):

        # Pretending we are hard at work
        time.sleep(0.1)

        # Validate input
        # If this was a consumer then incoming data MUST be checked

        # Increment counter
        if (self.currentCount > self.maxCount + self.numberToAdd):
            self.currentCount = self.numberToAdd
        else:
            self.currentCount += 1

        return True, self.currentCount
