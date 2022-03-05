from ..commsInterface.commsInterface import XBeeInterface
import struct

DEVICE_PORT = 0


class GroundSendRecieve:
    def read(self, data):
        output1, output2, output3, output4, pitch, yaw, roll, status_display, grabber_pos = struct.unpack(
            'IIIIfffII', data
        )
        decoded_message = {
            'outputs': [output1, output2, output3, output4],
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'status_display': status_display,
            'grabber_pos': grabber_pos
        }
        self.decoded_q.append(decoded_message)

    def __init__(self):
        self.com = XBeeInterface()
        self.com.create_device(self.read, DEVICE_PORT)
        self.decoded_q = []

    def encode(self, request):
        encode_bytearray(request)

    def send(self, data):
        self.com.write(0, data)


def encode_bytearray(request):
    info = request['data']['info']
    date = request['data']['date']
    time = request['data']['time']

    def pack_to_chararray(len, val):
        return struct.pack(f"{len}s", val)

    byte_list = [
        pack_to_chararray(100, info),
        pack_to_chararray(6, date),
        pack_to_chararray(4, time)
    ]

    return bytes(b''.join(byte_list))
