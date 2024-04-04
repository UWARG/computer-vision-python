"""
Logs debug messages.
"""

import datetime
import inspect
import logging
import pathlib
import os
import types
import typing
import yaml

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

        # Open config file.
        try:
            with CONFIG_FILE_PATH.open("r", encoding="utf8") as file:
                try:
                    config = yaml.safe_load(file)
                except yaml.YAMLError as exc:
                    print(f"Error parsing YAML file: {exc}")
                    return -1
        except FileNotFoundError:
            print(f"File not found: {CONFIG_FILE_PATH}")
            return False, None
        except IOError as exc:
            print(f"Error when opening file: {exc}")
            return False, None

        # Get the path to the logs directory.
        log_directory_path = config["log_directory_path"]
        entries = os.listdir(log_directory_path)
        log_names = [
            entry for entry in entries if os.path.isdir(os.path.join(log_directory_path, entry))
        ]

        # Find the log directory for the current run, which is the most recent timestamp
        datetime_format = "%Y-%m-%d_%H-%M-%S"
        log_path = max(
            log_names,
            key=lambda datetime_string: datetime.datetime.strptime(
                datetime_string, datetime_format
            ),
        )

        filename = f"{log_directory_path}/{log_path}/{name}.log"

        # Formatting configurations for the logger
        file_handler = logging.FileHandler(filename=filename, mode="w")  # Handles logging to file.
        stream_handler = logging.StreamHandler()  # Handles logging to terminal.

        formatter = logging.Formatter(
            fmt="%(asctime)s: [%(levelname)s] %(message)s",
            datefmt="%I:%M:%S",
        )

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Create a unique logger instance and configure it
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
    def message_and_metadata(message: str, frame: typing.Optional[types.FrameType]) -> str:
        """
        Extracts metadata from frame and appends it to the message.
        """
        function_name = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_number = inspect.getframeinfo(frame).lineno

        return f"[{filename} | {function_name} | {line_number}] {message}"

    def debug(self, message: str, frame: typing.Optional[types.FrameType]) -> None:
        """
        Logs a debug level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.debug(message)

    def info(self, message: str, frame: typing.Optional[types.FrameType]) -> None:
        """
        Logs an info level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.info(message)

    def warning(self, message: str, frame: typing.Optional[types.FrameType]) -> None:
        """
        Logs a warning level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.warning(message)

    def error(self, message: str, frame: typing.Optional[types.FrameType]) -> None:
        """
        Logs an error level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.error(message)

    def critical(self, message: str, frame: typing.Optional[types.FrameType]) -> None:
        """
        Logs a critical level message.
        """
        message = self.message_and_metadata(message, frame)
        self.logger.critical(message)
