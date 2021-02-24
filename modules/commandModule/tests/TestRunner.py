import os, unittest
from datetime import datetime
import logging

if __name__ == "__main__":
    """
    Test suite to be run using the following command in CLI:
    $ python3 -m commandModule.tests.TestRunner
    note: needs to be run from the 'modules' folder
    """

    # set up logging
    loggerFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testLogs", "debug.log")
    open(loggerFile, "w").close()
    logging.basicConfig(filename=loggerFile,
                        filemode="a",
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # get full test suite using discovery method
    testDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testCases")
    topDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    testSuite = unittest.TestLoader().discover(start_dir=testDir, pattern="test*.py", top_level_dir=topDir)

    # create log file
    currentDateTime = datetime.today().strftime("%H%M%S_%Y_%m_%d")
    logFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testLogs", "{}".format(currentDateTime))
    f = open(logFile, "w") 

    # run test and output results to log file
    runner = unittest.TextTestRunner(f)
    unittest.main(verbosity=2, defaultTest="testSuite", testRunner=runner)