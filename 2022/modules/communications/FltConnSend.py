import struct

from modules.commsInterface.commsInterface import UARTInterface

UART_PORT = ""
PADDING_BYTES = 5


class FltConnSend:
    def __init__(self):
        self.com = UARTInterface(UART_PORT)
        self.endpoint = self.com.create_end_point_FC()

    @staticmethod
    def build_fijo_for_qr(request):
        return build_fijo_bytearray(request)

    def send_fijo(self, fijo):
        self.com.write(endpointId=self.endpoint, data=fijo)
        return True


# Making this its own function so we can test it alone
def build_fijo_bytearray(request):
    latitude = request['data']['latitude']
    longitude = request['data']['longitude']

    qr_scan_flag = request['data']['qr_scan_flag']
    detect_flag = request['data']['detect_flag']
    takeoff_command = request['data']['takeoff_command']

    def pack_to_int(val):
        return struct.pack("i", val)

    def pack_to_float(val):
        return struct.pack("f", val)

    byte_list = [
        struct.pack("c", "$"[0].encode("ascii")),
        pack_to_int(detect_flag),
        pack_to_int(qr_scan_flag),
        pack_to_int(takeoff_command),
        *[struct.pack("x") for _ in range(PADDING_BYTES)],
        pack_to_float(latitude),
        pack_to_float(longitude)
    ]

    return bytes(b''.join(byte_list))
