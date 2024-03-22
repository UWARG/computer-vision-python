import inspect
import linecache

def message_and_metadata(message, frame):
    function_name = frame.f_code.co_name
    filename = frame.f_code.co_filename
    line_number = inspect.getframeinfo(frame).lineno

    return f'[{filename} | {function_name} | {line_number}] {message}'