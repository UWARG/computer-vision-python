import multiprocessing as mp
import time

"""
This is an example of a multiprocessed producer
A producer-consumer will have both an input and output, shouldn't be too hard
"""

class Counter:
    """
    Increments its internal counter
    Producer: Outputs current count
    """

    # They're all public in this example, but should probably be private
    # Globals must be initialized here ("compile"-time)
    # They could be initialized at runtime before starting the process but this is HIGHLY DISCOURAGED
    maxCount = 999


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
