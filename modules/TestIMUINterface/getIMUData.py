import serial
import time

class getIMUINterface:
    def getIMUData(comPort):
        ser = serial.Serial(comPort, 9600)
        while(True):
            b = ser.readline()
            x = 0
            y = 0
            z = 0
            timestamp = 0
            for i in range(16):
                x = x + (2**i)*(b&(1<<i)) #b&(1<<i) is the bit at the ith position. multiply by 2^i for decimal value, and add to total

            for i in range(17, 33):
                y = y + (2**i)*(b&(1<<i))

            for i in range(34, 50):
                z = z + (2**i)*(b&(1<<i))

            for i in range(50, 83):
                timestamp = timestamp + (2**i)*(b&(1<<i))

            output = {
                "x":x,
                "y":y,
                "z":z,
                "timestamp": timestamp
            }

            return output
