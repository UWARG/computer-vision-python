import time
import getIMUData


if __name__ == "__main__":

    getIMUWorker = getIMUData.getIMUInterface("COM6")

    while (True):
        print(getIMUWorker.getIMUData())
