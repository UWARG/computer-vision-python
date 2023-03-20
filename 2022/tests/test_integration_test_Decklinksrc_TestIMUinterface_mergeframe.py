import pytest
import time

from modules.decklinksrc.decklinksrc import DeckLinkSRC
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.TestIMUInterface.getIMUData import getIMUInterface
from modules.timestamp.timestamp import Timestamp

@pytest.mark.skip(reason="requires physical hardware")
def test_decklink_IMUinterface_to_mergeframe():
    # Setup
    DeckLinkSrc = DeckLinkSRC()
    Interface = getIMUInterface()
    comPort = "6"
    Merger = MergeImageWithTelemetry()

    currFrame = Timestamp(DeckLinkSrc.grab())
    Merger.set_image(currFrame)
    # At least 2 different telemetry data required
    currData1 = Timestamp(dict())#Interface.getIMUData(comPort))
    time.sleep(0.1)
    currData2 = Timestamp(dict())#Interface.getIMUData(comPort))
    Merger.put_back_telemetry(currData1)
    Merger.put_back_telemetry(currData2)

    # Run
    currMergedData = Merger.get_closest_telemetry()

    # Test
    assert currMergedData[0] == True
    assert currMergedData[1] != None
    assert currMergedData[1].image.any() != None
    assert currMergedData[1].telemetry != None
