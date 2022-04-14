import pytest

from modules.decklinksrc.decklinksrc import DeckLinkSRC
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.TestIMUINterface.getIMUData import getIMUINterface

def test_decklink_IMUinterface_to_mergeframe():
        DeckLinkSrc = DeckLinkSRC()
        Interface = getIMUINterface()
        Merger = MergeImageWithTelemetry()
        currFrame = DeckLinkSrc.grab()
        currMergedData = Merger.get_closest_telemetry()
                
        assert Interface.getIMUData != None
        assert currFrame.any() != None
        assert currMergedData[0] == True
        assert currMergedData[1] != None
