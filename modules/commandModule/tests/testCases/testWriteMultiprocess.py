from ...commandModule import CommandModule
import unittest
import json
import os
import multiprocessing
from time import time
from datetime import datetime

"""
NOTE
This is supposed to be an integration test, but with no proper integration testing framework, I'm just doing it here for now
Remember to remove this before pushing
"""

blah = CommandModule()

def runOne(blah):
    blah.set_begin_takeoff(beginTakeoff=False)

def runTwo(blah):
    print(id(blah))
    blah.set_begin_landing(beginLanding=True)

class TestCaseWritingMultiprocessToPIGOFile(unittest.TestCase):
    """
    Test Case: Multiprocess write write to PIGO file results in pass
    Methods to test:
    - set_gps_coordintes
	- set_ground_commands
	- set_gimbal_commands
	- set_begin_landing
	- set_begin_takeoff
	- set_disconnect_autopilot
    """

    def setUp(self):
        self.synchronizer = multiprocessing.Barrier(2)
        self.serializer = multiprocessing.Lock()
        self.testData= {"beginLanding": True,
                        "beginTakeoff": False}
        self.pigoFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "testJSONs", "test.json")
        self.commandModule = CommandModule(pigoFileDirectory=self.pigoFile)
        blah = self.commandModule
        print(id(self.commandModule))

    def tearDown(self):
        open(self.pigoFile, "w").close() # delete file contents before next unit test

    def __read_json(self):
        # used to check file output
        with open(self.pigoFile, "r") as pigoFile:
            results = json.load(pigoFile)
        return results

    def test(self):
        self.ArgOne = True
        self.ArgTwo = False
        multiprocessing.Process(target=self.commandModule.set_begin_landing, args=(self.ArgOne,)).start()
        multiprocessing.Process(target=self.commandModule.set_begin_takeoff, args=(self.ArgTwo,)).start()
        self.assertEqual(self.__read_json(), self.testData)