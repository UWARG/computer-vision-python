from modules.commsInterface.commsInterface import UARTInterface

UART_PORT = ""


class FltConnSend:
    def __init__(self):
        self.com = UARTInterface(UART_PORT)
        self.endpoint = self.com.create_end_point_FC()

    @staticmethod
    def build_fijo_for_qr(request):
        latitude = request['data']['latitude']
        longitude = request['data']['longitude']

        return f"$FIJO;1;1;0;{latitude};{longitude}"

    def send_fijo(self, fijo):
        self.com.write(endpointId=self.endpoint, data=fijo)
        return True
