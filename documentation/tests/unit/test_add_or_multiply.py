"""
Pytest example. To run:
```
cd documentation/tests/
pytest
```
"""

import math

import pytest

import add_or_multiply


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


# Pytest fixtures are reusable setup components
# Makes it easier when setup is complicated
@pytest.fixture()
def adder() -> add_or_multiply.AddOrMultiply:  # type: ignore
    """
    Creates AddOrMultiply in addition state.
    """
    add = add_or_multiply.AddOrMultiply(add_or_multiply.MathOperation.ADD)
    yield add  # type: ignore


@pytest.fixture()
def multiplier() -> add_or_multiply.AddOrMultiply:  # type: ignore
    """
    Creates AddOrMultiply in multiplication state.
    """
    multiply = add_or_multiply.AddOrMultiply(add_or_multiply.MathOperation.MULTIPLY)
    yield multiply  # type: ignore


class TestAddition:
    """
    Unit tests can be organized into groups under classes.
    The function or method name contains test somewhere for Pytest to run it.
    """

    def test_add_positive(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add 2 positive numbers.
        The parameter names must match the fixture function name to be used.
        """
        # Setup
        # Hardcode values on the right side if possible
        input_1 = 1.2
        input_2 = 3.4
        # expected is the variable name to compare against
        expected = 4.6

        # Run
        # actual is the variable name to compare
        actual = adder.add_or_multiply(input_1, input_2)

        # Test
        # Pytest unit tests pass if there are no unexpected exceptions
        # Actual before expected
        # Use math.isclose() for floating point comparison
        assert math.isclose(actual, expected)

    def test_add_large_positive(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add 2 positive numbers.
        """
        # Setup
        input_1 = 400_000_000.0
        input_2 = 0.1
        expected = 400_000_000.1

        # Run
        actual = adder.add_or_multiply(input_1, input_2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_positive_negative(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add positive and negative number.
        """
        # Setup
        input_1 = 0.1
        input_2 = -1.0
        expected = -0.9

        # Run
        actual = adder.add_or_multiply(input_1, input_2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_positive_and_same_negative(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add positive and negative number.
        """
        # Setup
        input_1 = 1.5
        input_2 = -1.5
        expected = 0.0

        # Run
        actual = adder.add_or_multiply(input_1, input_2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_negative(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add positive and negative number.
        """
        # Setup
        input_1 = -0.5
        input_2 = -1.0
        expected = -1.5

        # Run
        actual = adder.add_or_multiply(input_1, input_2)

        # Test
        assert math.isclose(actual, expected)

    # There are many more tests that can be made:
    # * Large positive with large positive
    # * Large positive with small negative
    # * Large negative with small negative
    # * Large negative with large negative


class TestMultiply:
    """
    Many multiplication cases need to be covered as well.
    """

    def test_multiply_positive(self, multiplier: add_or_multiply.AddOrMultiply) -> None:
        """
        Multiply 2 positive numbers.
        Different fixture so different parameter name.
        """
        # Setup
        input_1 = 1.2
        input_2 = 3.4
        expected = 4.08

        # Run
        actual = multiplier.add_or_multiply(input_1, input_2)

        # Test
        assert math.isclose(actual, expected)


class TestSwap:
    """
    Test a different method.
    """

    def test_swap_add_to_multiply(self, adder: add_or_multiply.AddOrMultiply) -> None:
        """
        Add and then multiply.
        """
        # Setup
        expected = add_or_multiply.MathOperation.MULTIPLY

        # Run
        adder.swap_state()

        # Test
        # Better to test private members directly rather than rely on other class methods
        # since the more dependencies a test has the less unit and independent it is
        actual = adder._AddOrMultiply__operator  # type: ignore

        assert actual == expected

    def test_swap_multiply_to_add(self, multiplier: add_or_multiply.AddOrMultiply) -> None:
        """
        Multiply and then add.
        """
        # Setup
        expected = add_or_multiply.MathOperation.ADD

        # Run
        multiplier.swap_state()

        # Test
        actual = multiplier._AddOrMultiply__operator  # type: ignore

        assert actual == expected
