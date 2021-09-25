import usb.core
import usb.util

class USBInterface:
    def create_end_point_FC(self):
        """
        Returns new endpoint either a serial object (UART) or endpoint object (USB)
        that points to flight controller
        """
        self.dev = usb.core.find()

        if self.dev is None:
            raise ValueError('Device Not Found')

        self.ep = self.dev[0].interfaces()[0].endpoints()[0]
        self.dev.set_configuration()

        return self.ep.bEndpointAddress

    def read(self, endpointId):
        """
        Issues read command in either serial (UART) or pyusb (USB) using endpointId.
        endpointId could be serial port or usb endpoint.
        """

        return self.dev.read(endpointId, self.ep.wMaxPacketSize)
        

    def write(self, endpointId, data):
        """
        Issues write command in either serial (UART) or pyusb (USB).
        """
        
        self.dev.write(endpointId, data)
        return True

