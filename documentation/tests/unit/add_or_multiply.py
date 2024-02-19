"""
Simple class for Pytest example.
"""

import enum


class MathOperation(enum.Enum):
    """
    Enumeration for Add or Multiply.
    """

    ADD = 0
    MULTIPLY = 1


class AddOrMultiply:
    """
    Add or multiply depending on state.
    """

    def __init__(self, switch: MathOperation):
        self.__operator = switch

    def add_or_multiply(self, num1: float, num2: float) -> float:
        """
        Adds or multiplies numbers based on internal state.
        """
        if self.__operator == MathOperation.ADD:
            return num1 + num2

        if self.__operator == MathOperation.MULTIPLY:
            return num1 * num2

        raise NotImplementedError

    def swap_state(self) -> None:
        """
        Swaps internal state.
        """
        if self.__operator == MathOperation.ADD:
            self.__operator = MathOperation.MULTIPLY
        elif self.__operator == MathOperation.MULTIPLY:
            self.__operator = MathOperation.ADD
        else:
            raise NotImplementedError
