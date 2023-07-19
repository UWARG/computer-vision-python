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
    pass


def test_expect_exception():
    """
    If an exception is expected.
    """
    with pytest.raises(Exception):
        x = 1 / 0
