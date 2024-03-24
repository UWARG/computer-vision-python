"""
Pytest example. To run:
```
cd documentation/tests/
pytest
```
"""

import pytest


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


def test_trivial_pass() -> None:
    """
    Unit test will pass by default.
    """
    # Keyword used to indicate end of empty function
    # pylint: disable-next=unnecessary-pass
    pass


def test_1_plus_1_equals_2() -> None:
    """
    Easy unit test.
    """
    expected = 2
    actual = 1 + 1

    assert actual == expected


def test_expect_exception() -> None:
    """
    If an exception is expected.
    """
    with pytest.raises(Exception):
        _ = 1 / 0
