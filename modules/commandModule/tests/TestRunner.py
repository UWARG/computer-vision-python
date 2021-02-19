import os, unittest
from datetime import datetime
import importlib

if __name__ == "__main__":
    """
    Test suite to be run using the following command in CLI:
    $ python3 -m commandModule.tests.TestRunner
    note: needs to be run from the 'modules' folder
    """

    # get full test suite using discovery method
    testDir = str(__file__).replace("/TestRunner.py", "") + "/testCases"
    topDir = str(__file__).replace("/commandModule/tests/TestRunner.py", "")
    testSuite = unittest.TestLoader().discover(start_dir=testDir, pattern="test*.py", top_level_dir=topDir)

    # create log file
    currentDateTime = datetime.today().strftime("%H:%M:%S_%Y_%m_%d")
    logFile = str(__file__).replace("/TestRunner.py", "") + "/testLogs/{0}.log".format(currentDateTime)
    f = open(logFile, "w") 

    # run test and output results to log file
    runner = unittest.TextTestRunner(f)
    unittest.main(verbosity=2, defaultTest="testSuite", testRunner=runner)