"""
Logs debug messages.
"""

import datetime
import logging
import pathlib
import os
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
        datetime_format = "%Y-%m-%d_%H:%M:%S"
        log_path = max(
            [
                datetime.datetime.strptime(datetime_string, datetime_format)
                for datetime_string in log_names
            ]
        ).strftime(datetime_format)
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
        print(logger.type)

        return True, Logger(cls.__create_key, logger)

    def __init__(self, class_create_private_key: object, logger: logging.Logger) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_create_private_key is Logger.__create_key, "Use create() method."

        self.logger = logger
