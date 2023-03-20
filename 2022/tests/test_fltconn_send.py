from modules.communications.FltConnSend import build_fijo_bytearray


def test_fltConnEncoding():
    command = {
        "type": "QR_COORDINATES",
        "data": {
            "longitude": 1.23,
            "latitude": 1.23,
            "qr_scan_flag": 1,
            "takeoff_command": 0,
            "detect_flag": 1
        }
    }
    byte_array = build_fijo_bytearray(command)
    print(byte_array)
    print(len(byte_array))

