import time

"""
This is an example of a multiprocessed producer-consumer
"""

class Printer:
    """
    Concatenates a prefix to the input
    Consumer: Gets a value from the provided queue
    Producer: Puts the result into the provided queue
    """

    # They're all public in this example, but should probably be private
    # Globals must be initialized here ("compile"-time)
    # They could be initialized at runtime before starting the process but this is HIGHLY DISCOURAGED
    prefixStart = "PrinterGlobal "


    def __init__(self, prefix):

        self.prefix = self.prefixStart + prefix


    # The working function
    def print(self, inData):

        # Pretending we are hard at work
        time.sleep(0.1)

        # Validate input
        # Incoming data MUST be checked in this class,
        # not the wrapper function!
        # It should be more sophisticated than this,
        # probably something like class method for checking and unpacking
        if (inData == None):
            return False, None
        suffix = str(inData)

        # Print string
        stringToPrint = self.prefix + suffix

        # Outgoing data MUST be in this form
        # so that bad data isn't going in the queue
        # Data must be packed into one variable before going out to the wrapper function!
        return True, stringToPrint
