import inspect
import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def message_and_metadata(message, frame):
    function_name = frame.f_code.co_name
    filename = frame.f_code.co_filename
    line_number = inspect.getframeinfo(frame).lineno

    return f"[{filename} | {function_name} | {line_number}] {message}"


def log_message(message, level, frame, queue):
    queue.queue.put((message_and_metadata(message, frame), level), block=False)
