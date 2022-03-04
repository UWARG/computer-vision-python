import unittest
import numpy as np

from modules.framePreProc import framePreProc
class TestFilter(unittest.TestCase):

    """
    Tests FramePreProc.filter
    """

    def setUp(self):
        eulerDictLast = {'yaw': 30,
                'pitch': 45, 'roll': 15}
        self.framePreProc = framePreProc.FramePreProc(eulerDictLast)
        return

    def test_empty_dict(self):
        self.framePreProc.eulerDictLast = None
        inThreshold = {'yaw': 35, 'pitch': 40, 'roll': 10}
        result = self.framePreProc.filter(inThreshold)
        self.assertFalse(result)

    def test_within_threshold(self):
        inThreshold = {'yaw': 35, 'pitch': 40, 'roll': 10}
        result = self.framePreProc.filter(inThreshold)
        self.assertTrue(result)

    def test_outside_threshold(self):
        outThreshold = {'yaw': 90, 'pitch': 45, 'roll': 15}
        result = self.framePreProc.filter(outThreshold)
        self.assertFalse(result)
