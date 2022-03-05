from modules.communications.groundSend import encode_bytearray


def test_groundConnEncoding():
    command = {
        "type": "GROUND",
        "data": {
            "info": "abcdef",
            "date": "11032002",
            "time": "1135"
        }
    }
    byte_array = encode_bytearray(command)
    print(byte_array)
    print(len(byte_array))

