"""
Example of an intermediate struct representation.
"""


# This class is just a struct containing some members
# pylint: disable-next=too-few-public-methods
class IntermediateStruct:
    """
    Example of a simple struct.
    """

    def __init__(self, number: int, sentence: str):
        """
        Constructor.
        """
        self.number = number
        self.sentence = sentence
