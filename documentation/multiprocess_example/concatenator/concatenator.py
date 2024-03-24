"""
Contains the Concatenator class.
"""

import time

from .. import intermediate_struct


class Concatenator:
    """
    Concatenates a prefix and suffix to the object.
    """

    def __init__(self, prefix: str, suffix: str) -> None:
        """
        Constructor sets the prefix and suffix.
        """
        self.__prefix = prefix
        self.__suffix = suffix

    # The working function
    def run_concatenation(
        self, middle: intermediate_struct.IntermediateStruct
    ) -> "tuple[bool, str]":
        """
        Concatenate the prefix and suffix to the input.
        """
        # The class is responsible for unpacking the intermediate type
        # Validate input
        input_number = middle.number
        input_string = middle.sentence
        if input_string == "":
            # Function returns result and the output
            return False, ""

        # Print string
        concatenated_string = self.__prefix + str(input_number) + self.__suffix

        # Pretending this is hard at work
        time.sleep(0.1)

        # Function returns result and the output
        return True, concatenated_string
