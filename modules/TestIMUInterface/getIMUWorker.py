import time
import getIMUData


if __name__ == "__main__":

    getIMUWorker = getIMUData.getIMUInterface("COM6")

    for i in range(0, 1000):
        xyz = getIMUWorker.getIMUData()
        print(xyz)
        ZYX = getIMUWorker.ZYXFromIMU(xyz)
        print(ZYX)

    # Performance gets very bad when near poles (completely up or completely down)
    # Otherwise pretty good!
    getIMUWorker.performance()
