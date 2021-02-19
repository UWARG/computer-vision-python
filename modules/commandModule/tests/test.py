import os, sys
sys.path.insert(0, os.path.abspath(".."))

import commandModule
import unittest
import logging

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    test = commandModule.CommandModule("test.json", "test.json")

    b = {"heading": 1.234, "latestDistance": 2.0}
    test.set_gps_coordinates(b)

    test.set_gps_coordinates(b)
    test.set_PIGO_directory(None)