"""
Contains the AddRandom class.
"""

import time
import random

from modules.common.modules.logger import logger
from .. import intermediate_struct


class AddRandom:
    """
    Adds a random number to the input.

    A new random number is generated every `__add_change_count` times.
    """

    def __init__(
        self, seed: int, max_random_term: int, add_change_count: int, local_logger: logger.Logger
    ) -> None:
        """
        Constructor seeds the RNG and sets the max add and
        number of adds before a new random number is chosen.
        """
        random.seed(seed)
        # Maximum value that can be added
        self.__max_random_term = max_random_term

        # Number of adds before getting a new random number
        self.__add_change_count = add_change_count

        self.__current_random_term = self.__generate_random_number(0, self.__max_random_term)
        self.__add_count = 0

        self.__logger = local_logger

    @staticmethod
    def __generate_random_number(min_value: int, max_value: int) -> int:
        """
        Generates a random number between min and max, inclusive.
        """
        return random.randrange(min_value, max_value + 1)

    def run_add_random(self, term: int) -> "tuple[bool, intermediate_struct.IntermediateStruct]":
        """
        Adds a random number to the input and returns the sum.
        """
        # Log
        self.__logger.debug("Run", True)

        add_sum = term + self.__current_random_term

        # Change the random term if the add count has been reached
        self.__add_count += 1
        if self.__add_count >= self.__add_change_count:
            self.__current_random_term = self.__generate_random_number(0, self.__max_random_term)
            self.__add_count = 0

        # Pretending this class is hard at work
        time.sleep(0.2)

        add_string = ""
        if add_sum % 2 == 0:
            add_string = "even"

        output = intermediate_struct.IntermediateStruct(add_sum, add_string)

        # Function returns result and the output
        # The class is responsible for packing the intermediate type
        return True, output
