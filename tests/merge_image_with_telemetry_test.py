import pytest
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.timestamp.timestamp import Timestamp
def testMerging(): 
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    telemetry1 = Timestamp(["telemetry1"])
    telemetry2 = Timestamp(["telemetry2"])
    
    merger.put_back(telemetry1)
    merger.put_back(telemetry2)

    [success, output] = merger.merge_with_closest_telemetry(image.timestamp, image.data)
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry1.data]

if __name__ == "__main__": 
    testMerging()

