"""
Pytest example. To run:
```
cd documentation/tests/
pytest
```
"""

import pytest


def test_trivial_pass():
    """
    Unit test will pass by default.
    """
    # Keyword used to indicate end of empty function
    # pylint: disable-next=unnecessary-pass
    pass


def test_1_plus_1_equals_2():
    """
    Easy unit test.
    """
    expected = 2
    actual = 1 + 1

    assert actual == expected


def test_expect_exception():
    """
    If an exception is expected.
    """
    with pytest.raises(Exception):
        _ = 1 / 0
