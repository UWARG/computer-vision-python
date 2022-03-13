"""
Unit tests for searchExplosive module
"""

import pytest
from modules.searchExplosive.searchExplosive import SearchExplosive

def test_empty_frame():
    detector = SearchExplosive(None)
    result = detector.edge_detection()
    assert not result
