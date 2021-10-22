import unittest
import xmlrunner
from tests import *

if __name__ == "__main__":
    with open('tests/report.xml', 'wb') as output:
        unittest.main(testRunner=xmlrunner.XMLTestRunner(output=output), failfast=False,buffer=False,catchbreak=False)