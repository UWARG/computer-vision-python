"""
ZP message and timestamp
"""
import time


# Basically a struct
# pylint: disable=too-few-public-methods
class MessageAndTime:
    """
    Contains ZP message and timestamp
    """
    __create_key = object()

    @classmethod
    def create(cls, message) -> "tuple[bool, MessageAndTime | None]":
        """
        message: 1 of TelemMessages. No type annotation due to several possible types
        """
        if len(message) == 0:
            return False, None

        return True, MessageAndTime(cls.__create_key, message)
    
    def __init__(self, class_private_create_key, message):
        """
        Private constructor, use create() method
        Constructor sets timestamp to current time
        """
        assert class_private_create_key is MessageAndTime.__create_key, "Use create() method"

        self.message = message
        self.timestamp = time.time()

# pylint: enable=too-few-public-methods
