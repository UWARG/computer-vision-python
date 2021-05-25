import pytest
import cv2
from modules.targetAcquisition.targetAcquisition import TargetAcquisition
from modules.decklinksrc.decklinksrc import DeckLinkSRC

def test_decklink_targetAcquisition():
    decklinkSrc = DeckLinkSRC()
    tracker = TargetAcquisition()
    while True:
        
        curr_frame = decklinkSrc.grab()
        assert curr_frame != None
        decklinkSrc.display()
        tracker.set_curr_frame(curr_frame)
        assert tracker.get_coordinates() != None
    