import time
from .getIMUData import getIMUINterface

getIMUWorker = getIMUINterface()

while(True):
    getIMUWorker.getIMUData() #pass in comport
    time.sleep(5)