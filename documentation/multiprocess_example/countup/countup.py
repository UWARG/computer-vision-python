"""
Contains the Countup class.
"""

import inspect
import time

from modules.logger import logger


class Countup:
    """
    Increments its internal counter and outputs current counter.
    """

    def __init__(
        self, start_thousands: int, max_iterations: int, local_logger: logger.Logger
    ) -> None:
        """
        Constructor initializes the start and max points.
        """
        self.__start_count = start_thousands * 1000
        self.__max_count = self.__start_count + max_iterations
        self.__current_count = self.__start_count

        self.__logger = local_logger

    def run_countup(self) -> "tuple[bool, int]":
        """
        Counts upward.
        """
        # Log
        frame = inspect.currentframe()
        self.__logger.debug("Run", frame)

        # Increment counter
        self.__current_count += 1
        if self.__current_count > self.__max_count:
            self.__current_count = self.__start_count

        # Pretending to be hard at work
        time.sleep(0.15)

        # Function returns result and the output
        return True, self.__current_count
