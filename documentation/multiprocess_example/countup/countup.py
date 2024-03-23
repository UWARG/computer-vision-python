"""
Contains the Countup class.
"""

import time


class Countup:
    """
    Increments its internal counter and outputs current counter.
    """

    def __init__(self, start_thousands: int, max_iterations: int) -> None:
        """
        Constructor initializes the start and max points.
        """
        self.__start_count = start_thousands * 1000
        self.__max_count = self.__start_count + max_iterations
        self.__current_count = self.__start_count

    def run_countup(self) -> "tuple[bool, int]":
        """
        Counts upward.
        """
        # Increment counter
        self.__current_count += 1
        if self.__current_count > self.__max_count:
            self.__current_count = self.__start_count

        # Pretending to be hard at work
        time.sleep(0.15)

        # Function returns result and the output
        return True, self.__current_count
