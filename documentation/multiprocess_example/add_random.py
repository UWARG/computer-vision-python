"""
Contains the AddRandom class.
"""
import time
from typing import Tuple

import random

import intermediate_struct


# This class does very little, but still has state
# pylint: disable=too-few-public-methods
class AddRandom:
    """
    Adds a random number to the input.

    A new random number is generated every `__ADD_SWITCH_COUNT` times.
    """
    def __init__(self, seed: int, max_random_term: int, add_change_count: int):
        """
        Constructor seeds the RNG and sets the max add and
        number of adds before a new random number is chosen.
        """
        random.seed(seed)
        # Maximum value that can be added
        # Constant within class does not follow Pylint naming
        # pylint: disable=invalid-name
        self.__MAX_RANDOM_TERM = max_random_term
        # pylint: enable=invalid-name

        # Number of adds before getting a new random number
        # Constant within class does not follow Pylint naming
        # pylint: disable=invalid-name
        self.__ADD_CHANGE_COUNT = add_change_count
        # pylint: enable=invalid-name

        self.__current_random_term = self.__generate_random_number(0, self.__MAX_RANDOM_TERM)
        self.__add_count = 0

    @staticmethod
    def __generate_random_number(min_value: int, max_value: int) -> int:
        """
        Generates a random number between min and max, inclusive.
        """
        return random.randrange(min_value, max_value + 1)

    def run_add_random(self, term: int) -> Tuple[bool, intermediate_struct.IntermediateStruct]:
        """
        Adds a random number to the input and returns the sum.
        """
        add_sum = term + self.__current_random_term

        # Change the random term if the add count has been reached
        self.__add_count += 1
        if self.__add_count >= self.__ADD_CHANGE_COUNT:
            self.__current_random_term = self.__generate_random_number(0, self.__MAX_RANDOM_TERM)
            self.__add_count = 0

        # Pretending this class is hard at work
        time.sleep(0.2)

        add_string = ""
        if add_sum % 2 == 0:
            add_string = "even"

        # For some reason Pylint hates having more than 1 parameter in a constructor
        # pylint: disable=too-many-function-args
        output = intermediate_struct.IntermediateStruct(add_sum, add_string)
        # pylint: enable=too-many-function-args

        # Function returns result and the output
        # The class is responsible for packing the intermediate type
        return True, output

# pylint: enable=too-few-public-methods
