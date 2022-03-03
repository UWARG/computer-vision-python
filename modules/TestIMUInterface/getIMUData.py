import serial
import time
import math


class getIMUInterface:

    def __init__(self, comPort: str, baudrate: int = 115200):
        self.__ser = serial.Serial(comPort, baudrate)

        self.__totalAnglesCount = 0
        self.__diffGTOneTenth = 0
        self.__diffGTTwoTenths = 0
        self.__diffGTThreeTenths = 0

    def openPort(self, comPort: str, baudrate: int) -> None:
        self.__ser = serial.Serial(comPort, baudrate)

    def closePort(self) -> None:
        self.__ser.close()

    def getIMUData(self) -> dict:

        self.__ser.flush()
        b = self.__ser.readline()
        rawString = b.decode("ISO-8859-1")
        xyzString = rawString.split(" ")

        if (len(xyzString) < 3):
            return dict(x=0, y=0, z=0)

        x = int(xyzString[0])
        y = int(xyzString[1])
        z = int(xyzString[2])

        output = dict(
            x = x,
            y = y,
            z = z
        )

        return output

    def ZYXFromIMU(self, xyz: dict) -> dict:

        # Arbitrary normalization constant
        NORMAL = 17000
        x = xyz["x"]
        y = xyz["y"]
        z = xyz["z"]

        # q angle
        qInput = z / NORMAL
        if (qInput > 1):
            qAngle = 0
        elif (qInput < -1):
            qAngle = math.pi
        else:
            qAngle = math.acos(z / NORMAL)

        # p angle with x as input
        pInputX = x / (NORMAL * math.sin(qAngle))
        if (pInputX > 1):
            pAngleX = 0
        elif (pInputX < -1):
            pAngleX = math.pi
        else:
            pAngleX = math.acos(pInputX)

        # Quadrant
        # Remember inverse trig angle aliasing?
        if (y < 0):
            pAngleX = -pAngleX

        # p angle with y as input
        pInputY = y / (NORMAL * math.sin(qAngle))
        if (pInputY > 1):
            pAngleY = math.pi / 2
        elif (pInputY < -1):
            pAngleY = -math.pi / 2
        else:
            pAngleY = math.asin(pInputY)

        # Quadrant
        # Remember inverse trig angle aliasing?
        if (x < 0):
            pAngleY = -math.pi - pAngleY

        pDiff = abs(pAngleX - pAngleY)
        print(f"pAngleX: {math.degrees(pAngleX)}°, pAngleY: {math.degrees(pAngleY)}°, pDiff: {math.degrees(pDiff)}°")

        self.__totalAnglesCount += 1
        if (pDiff > 0.1):
            self.__diffGTOneTenth += 1

        if (pDiff > 0.2):
            self.__diffGTTwoTenths += 1

        if (pDiff > 0.3):
            self.__diffGTThreeTenths += 1

        output = dict(
            yaw = 0,
            pitch = math.pi / 2 - qAngle,
            roll = -1 * (pAngleX + pAngleY) / 2
        )

        return output

    def performance(self) -> None:
        print(f">0.1 rad: {self.__diffGTOneTenth / self.__totalAnglesCount}")
        print(f">0.2 rad: {self.__diffGTTwoTenths / self.__totalAnglesCount}")
        print(f">0.3 rad: {self.__diffGTThreeTenths / self.__totalAnglesCount}")
        print(f"Total: {self.__totalAnglesCount}")
