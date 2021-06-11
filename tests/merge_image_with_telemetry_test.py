import pytest
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.mergeImageWithTelemetry.mergedData import MergedData
from modules.timestamp.timestamp import Timestamp
def testMerging(): 
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    telemetry1 = Timestamp(["telemetry1"])
    telemetry2 = Timestamp(["telemetry2"])
    telemetry3 = Timestamp(["telemetry3"])

    merger.set_image(image)
    merger.put_back_telemetry(telemetry1)
    merger.put_back_telemetry(telemetry2)
    merger.put_back_telemetry(telemetry3)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry1.data]

    merger.set_image(image)
    [success, output] = merger.get_closest_telemetry()
    assert [success, output.telemetry] == [True, telemetry2.data]

    merger.set_image(image)
    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]
    # made it to an empty telemetry pipeline

if __name__ == "__main__": 
    testMerging()

