import pytest
import time
from datetime import datetime

from modules.mergeImageWithTelemetry.mergeImageWithTelemetry import MergeImageWithTelemetry
from modules.timestamp.timestamp import Timestamp

"""
    Unit tests for the MergeImageWithTelemetry module. 

    To run : 

        In the main directory, use the command 
            python -m tests.merge_image_with_telemetry_test
        should avoid the module not found errors (hopefully?)
"""

def test_empty():
    print("testing empty")
    merger = MergeImageWithTelemetry()
    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def test_no_image():
    print("testing no image")
    merger = MergeImageWithTelemetry()
    telemetry1 = Timestamp(["telemetry1"])
    telemetry2 = Timestamp(["telemetry2"])

    merger.put_back_telemetry(telemetry1)
    merger.put_back_telemetry(telemetry2)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def test_no_telemetry():
    print("testing no telemetry")
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    merger.set_image(image)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]

def test_first():
    print("testing first")
    merger = MergeImageWithTelemetry()
    image = Timestamp(["image"])
    telemetry1 = Timestamp(["telemetry1"])
    time.sleep(0.1)
    telemetry2 = Timestamp(["telemetry2"])

    merger.set_image(image)
    merger.put_back_telemetry(telemetry1)
    merger.put_back_telemetry(telemetry2)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry1.data]

def test_middle():
    print("testing middle")
    merger = MergeImageWithTelemetry()
    for i in range(4): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)
    time.sleep(0.1)

    image = Timestamp(["image"])
    merger.set_image(image)

    telemetry = Timestamp(["telemetry"])
    merger.put_back_telemetry(telemetry)

    time.sleep(0.1)
    for i in range(4, 9): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output.image, output.telemetry] == [True, image.data, telemetry.data]

def test_last():
    print("testing last")
    merger = MergeImageWithTelemetry()

    for i in range(4): 
        t = Timestamp([i])
        merger.put_back_telemetry(t)
    time.sleep(0.1)

    image = Timestamp(["image"])
    merger.set_image(image)

    telemetry = Timestamp(["telemetry"])
    merger.put_back_telemetry(telemetry)

    [success, output] = merger.get_closest_telemetry()
    assert [success, output] == [False, None]
