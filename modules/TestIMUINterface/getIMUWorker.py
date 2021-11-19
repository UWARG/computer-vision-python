import time
#from .getIMUData import getIMUINterface
import getIMUData

getIMUWorker = getIMUData.getIMUINterface('/dev/ttyACM0')

while(True):
    print(getIMUWorker.getIMUData()) 