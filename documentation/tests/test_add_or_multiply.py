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


# Pytest fixtures are reusable setup components
# Makes it easier when setup is complicated
@pytest.fixture()
def adder():
    """
    Creates AddOrMultiply in addition state.
    """
    add = add_or_multiply.AddOrMultiply(add_or_multiply.MathOperation.ADD)
    yield add

@pytest.fixture()
def multiplier():
    """
    Creates AddOrMultiply in multiplication state.
    """
    multiply = add_or_multiply.AddOrMultiply(add_or_multiply.MathOperation.MULTIPLY)
    yield multiply


class TestAddition:
    """
    Unit tests can be organized into groups under classes.
    The function or method name contains test somewhere for Pytest to run it.
    """
    def test_add_positive(self, adder: add_or_multiply.AddOrMultiply):
        """
        Add 2 positive numbers.
        The parameter names must match the fixture function name to be used.
        """
        # Setup
        # Hardcode values on the right side if possible
        input1 = 1.2
        input2 = 3.4
        # expected is the variable name to compare against
        expected = 4.6

        # Run
        # actual is the variable name to compare
        actual = adder.add_or_multiply(input1, input2)

        # Test
        # Pytest unit tests pass if there are no unexpected exceptions
        # Actual before expected
        # Use math.isclose() for floating point comparison
        assert math.isclose(actual, expected)

    def test_add_large_positive(self, adder: add_or_multiply.AddOrMultiply):
        """
        Add 2 positive numbers.
        """
        # Setup
        input1 = 400_000_000.0
        input2 = 0.1
        expected = 400_000_000.1

        # Run
        actual = adder.add_or_multiply(input1, input2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_positive_negative(self, adder: add_or_multiply.AddOrMultiply):
        """
        Add positive and negative number.
        """
        # Setup
        input1 = 0.1
        input2 = -1.0
        expected = -0.9

        # Run
        actual = adder.add_or_multiply(input1, input2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_positive_and_same_negative(self, adder: add_or_multiply.AddOrMultiply):
        """
        Add positive and negative number.
        """
        # Setup
        input1 = 1.5
        input2 = -1.5
        expected = 0.0

        # Run
        actual = adder.add_or_multiply(input1, input2)

        # Test
        assert math.isclose(actual, expected)

    def test_add_negative(self, adder: add_or_multiply.AddOrMultiply):
        """
        Add positive and negative number.
        """
        # Setup
        input1 = -0.5
        input2 = -1.0
        expected = -1.5

        # Run
        actual = adder.add_or_multiply(input1, input2)

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
    def test_multiply_positive(self, multiplier: add_or_multiply.AddOrMultiply):
        """
        Multiply 2 positive numbers.
        Different fixture so different parameter name.
        """
        # Setup
        input1 = 1.2
        input2 = 3.4
        expected = 4.08

        # Run
        actual = multiplier.add_or_multiply(input1, input2)

        # Test
        assert math.isclose(actual, expected)


class TestSwap:
    """
    Test a different method.
    """
    def test_swap_add_to_multiply(self, adder: add_or_multiply.AddOrMultiply):
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
        # Access required for test
        # pylint: disable=protected-access
        actual = adder._AddOrMultiply__operator
        # pylint: enable=protected-access
        assert actual == expected

    def test_swap_multiply_to_add(self, multiplier: add_or_multiply.AddOrMultiply):
        """
        Multiply and then add.
        """
        # Setup
        expected = add_or_multiply.MathOperation.ADD

        # Run
        multiplier.swap_state()

        # Test
        # Access required for test
        # pylint: disable=protected-access
        actual = multiplier._AddOrMultiply__operator
        # pylint: enable=protected-access
        assert actual == expected
