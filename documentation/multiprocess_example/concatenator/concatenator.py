"""
Contains the Concatenator class.
"""

import time

from modules.common.modules.logger import logger
from .. import intermediate_struct


class Concatenator:
    """
    Concatenates a prefix and suffix to the object.
    """

    def __init__(self, prefix: str, suffix: str, local_logger: logger.Logger) -> None:
        """
        Constructor sets the prefix and suffix.
        """
        self.__prefix = prefix
        self.__suffix = suffix

        self.__logger = local_logger

    # The working function
    def run_concatenation(
        self, middle: intermediate_struct.IntermediateStruct
    ) -> "tuple[bool, str]":
        """
        Concatenate the prefix and suffix to the input.
        """
        # Log
        self.__logger.debug("Run", True)

        # The class is responsible for unpacking the intermediate type
        # Validate input
        input_number = middle.number
        input_string = middle.sentence
        if input_string == "":
            # Function returns result and the output
            return False, ""

        # String to be printed
        concatenated_string = self.__prefix + str(input_number) + self.__suffix

        # Pretending this is hard at work
        time.sleep(0.1)

        # Function returns result and the output
        return True, concatenated_string
