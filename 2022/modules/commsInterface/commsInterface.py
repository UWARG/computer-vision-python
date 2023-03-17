import usb.core # import for pyusb
import usb.util # import for pyusb
import serial # import for pyserial
from digi.xbee.devices import XBeeDevice


class CommsInterface:
    def __init__(self, usb_type: int):
        """
        Initializes parameters for USB or UART

        Parameters
        ----------
        type: bool
            Boolean used to specify selection of USB or UART;
            0|False for USB, 1|True for UART
        """
        self.uart_or_usb = usb_type
    
    def create_end_point_FC(self):
        """
        Issues a read command in either serial (UART) or pyusb (USB).
        EndpointId can be the specific serial port to use or the usb endpoint to be used. 
        """
        if self.uart_or_usb == 0:
            if self.idVendor and self.idProduct is not None:
                self.dev = usb.core.find(idVendor=self.idVendor, idProduct=self.idProduct)
            else:
                self.dev = usb.core.find()
            if self.dev is None:
                raise ValueError('Device Not Found')

            self.ep = self.dev[0].interfaces()[0].endpoints()[0]
            self.dev.set_configuration()

            return self.ep.bEndpointAddress
        elif self.uart_or_usb == 1:
            return serial.Serial(self.uart_port, self.baudrate)

    def read(self, endpointId):
        """
        Issues a write command in either serial or pyusb.
        Data parameter should be a buffer. 
        """
        if self.uart_or_usb == 0:
            return self.dev.read(endpointId, self.ep.wMaxPacketSize)
        elif self.uart_or_usb == 1:
            read_data = endpointId.readline()
            return read_data

    def write(self, endpointId, data):
        """
        Returns a new endpoint, either a Serial object (UART) or Endpoint object (USB), that points to our flightcontroller.
        """
        if self.uart_or_usb == 0:
            self.dev.write(endpointId, data)
            return True
        elif self.uart_or_usb == 1:
            endpointId.write(data.encode())
            endpointId.close()
            return True


class USBInterface(CommsInterface):
    def __init__(self, idVendor=None, idProduct=None):
        # use lsusb -v in the terminal to list the idVendor and idProduct of connected usb devices 
        self.idVendor = idVendor
        self.idProduct = idProduct
        super().__init__(0)


class UARTInterface(CommsInterface):
    def __init__(self, uart_port: str, baudrate: int = 9600):
        self.uart_port = uart_port
        self.baudrate = baudrate
        super().__init__(1)


class XBeeInterface:
    def __init__(self):
        """
        Initializes a new XBee interface with an empty func_dict and device_dict
        """
        self.func_dict = dict()
        self.device_dict = dict()
        self.id_counter = 0

    def read_callback(self, device_id):
        def callback(xbee_message):
            data = xbee_message.data.decode("utf8")
            self.func_dict[device_id](data)
        return callback

    def write(self, device_id, data):
        self.device_dict[device_id].send_data_broadcast(data)

    def create_device(self, read_function, device_port):
        device = XBeeDevice(device_port, 9600)
        device.open()
        device.add_data_received_callback(self.read_callback(self.id_counter))
        self.func_dict[self.id_counter] = read_function
        self.device_dict[self.id_counter] = device
        self.id_counter += 1

