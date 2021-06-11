import pytest
from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.timestamp.timestamp import Timestamp

from datetime import datetime

"""
    Unit tests for the MergeImageWithTelemetry module. 

    To run : 

        In the main directory, use the command 
            python -m tests.merge_image_with_telemetry_test
        should avoid the module not found errors (hopefully?)
"""

def empty(): 
    merger = MergeImageWithTelemetry()
    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def no_image():
    merger = MergeImageWithTelemetry()
    telemetry1 = Timestamp(["telemetry1"])
    telemetry2 = Timestamp(["telemetry2"])

    merger.put_back_telemetry(telemetry1)
    merger.put_back_telemetry(telemetry2)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def no_telemetry(): 
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    merger.set_image(image)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def first(): 
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    telemetry1 = Timestamp(["telemetry1"])
    telemetry2 = Timestamp(["telemetry2"])

    merger.set_image(image)
    merger.put_back_telemetry(telemetry1)
    merger.put_back_telemetry(telemetry2)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry1.data]

def middle(): 
    merger = MergeImageWithTelemetry()
    for i in range(4): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)
    
    now = datetime.now()
    image = Timestamp(["image"], now)
    merger.set_image(image)

    telemetry = Timestamp(["telemetry"], now)
    merger.put_back_telemetry(telemetry)

    for i in range(4, 9): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry.data]

def last():
    merger = MergeImageWithTelemetry()

    for i in range(4): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)
    
    now = datetime.now()
    image = Timestamp(["image"], now)
    merger.set_image(image)

    telemetry = Timestamp(["telemetry"], now)
    merger.put_back_telemetry(telemetry)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]
    # made it to the end, the telemetry pipeline is waiting

if __name__ == "__main__": 
    print("testing empty")
    empty()
    print("****************** PASSED ************************")
    print("testing no image")
    no_image()
    print("****************** PASSED ************************")
    print("testing no telemetry")
    no_telemetry()
    print("****************** PASSED ************************")
    print("testing first")
    first()
    print("****************** PASSED ************************")
    print("testing middle")
    middle()
    print("****************** PASSED ************************")
    print("testing last")
    last()
    print("****************** PASSED ************************")