"""
Logs debug messages.
"""

import datetime
import inspect
import logging
import pathlib
import os

# Used in type annotation of logger parameters
# pylint: disable-next=unused-import
import types
import yaml

from utilities import yaml


CONFIG_FILE_PATH = pathlib.Path("config.yaml")


class Logger:
    """
    Instantiates Logger objects.
    """

    __create_key = object()

    @classmethod
    def create(cls, name: str) -> "tuple[bool, Logger | None]":
        """
        Create and configure a logger.
        """
        # Configuration settings
        result, config = yaml.open_yaml_file(CONFIG_FILE_PATH)
        if not result:
            print("ERROR: Failed to load configuration file")
            return False, None

        assert config is not None

        try:
            log_directory_path = config["logger"]["directory_path"]
            file_datetime_format = config["logger"]["file_datetime_format"]
            logger_format = config["logger"]["format"]
            logger_datetime_format = config["logger"]["datetime_format"]
        except KeyError:
            print("Config key(s) not found")
            return False, None

        # Get the path to the logs directory.
        entries = os.listdir(log_directory_path)

        if len(entries) == 0:
            print("ERROR: Must create a new log directory for this run")
            return False, None

        log_names = [
            entry for entry in entries if os.path.isdir(os.path.join(log_directory_path, entry))
        ]

        # Find the log directory for the current run, which is the most recent timestamp.
        log_path = max(
            log_names,
            key=lambda datetime_string: datetime.datetime.strptime(
                datetime_string, file_datetime_format
            ),
        )

        filepath = pathlib.Path(log_directory_path, log_path, f"{name}.log")

        # Formatting configurations for the logger.
        file_handler = logging.FileHandler(filename=filepath, mode="w")  # Handles logging to file.
        stream_handler = logging.StreamHandler()  # Handles logging to terminal.

        formatter = logging.Formatter(
            fmt=logger_format,
            datefmt=logger_datetime_format,
        )

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Create a unique logger instance and configure it.
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return True, Logger(cls.__create_key, logger)

    def __init__(self, class_create_private_key: object, logger: logging.Logger) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_create_private_key is Logger.__create_key, "Use create() method."

        self.logger = logger

    @staticmethod
    def message_and_metadata(message: str, frame: "types.FrameType | None") -> str:
        """
        Extracts metadata from frame and appends it to the message.
        """
        if frame is None:
            return f"[No frame data] {message}"

        assert frame is not None

        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_number = inspect.getframeinfo(frame).lineno

        return f"[{filename} | {function_name} | {line_number}] {message}"

    def debug(self, message: str, frame: "types.FrameType | None") -> None:
        """
        Logs a debug level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.debug(message)

    def info(self, message: str, frame: "types.FrameType | None") -> None:
        """
        Logs an info level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.info(message)

    def warning(self, message: str, frame: "types.FrameType | None") -> None:
        """
        Logs a warning level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.warning(message)

    def error(self, message: str, frame: "types.FrameType | None") -> None:
        """
        Logs an error level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.error(message)

    def critical(self, message: str, frame: "types.FrameType | None") -> None:
        """
        Logs a critical level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.critical(message)
