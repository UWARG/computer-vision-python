import serial
import time


class getIMUInterface:

    def __init__(self, comPort: str, baudrate: int = 115200):
        self.__ser = serial.Serial(comPort, baudrate)

    def openPort(self, comPort: str, baudrate: int) -> None:
        self.__ser = serial.Serial(comPort, baudrate)

    def closePort(self) -> None:
        self.__ser.close()

    def getIMUData(self) -> dict:

        self.__ser.flush()
        b = self.__ser.readline()
        rawString = b.decode("ISO-8859-1")
        xyzString = rawString.split(" ")

        x = int(xyzString[0])
        y = int(xyzString[1]) if (len(xyzString) > 1) else None
        z = int(xyzString[2]) if (len(xyzString) > 2) else None

        output = dict(
            x = x,
            y = y,
            z = z
        )

        return output
