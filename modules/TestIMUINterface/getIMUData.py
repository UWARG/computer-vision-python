import serial
import time
def bitwise_and_bytes(a, b):
    result_int = int.from_bytes(a, byteorder="big") & int.from_bytes(b, byteorder="big")
    return result_int.to_bytes(max(len(a), len(b)), byteorder="big")

class getIMUINterface:
    def __init__(self, comPort, baudrate=115200):
        self.ser = serial.Serial(comPort, baudrate)
    def getIMUData(self):

        self.ser.flush()
        b = self.ser.readline()
        b = b[:len(b)-2]

        x = 0
        y = 0
        z = 0
        coordinates = b.decode('ISO-8859-1').split(' ')

        output = {
            "x":coordinates[0],
            "y": coordinates[1] if len(coordinates)==3 else 0,
            "z":coordinates[2] if len(coordinates)==3 else 0
        }

        return output
