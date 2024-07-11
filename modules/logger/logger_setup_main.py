"""
Logger setup for `main()` .
"""

import datetime
import inspect
import pathlib

from . import logger


MAIN_LOGGER_NAME = "main"


def setup_main_logger(config: "dict") -> "tuple[bool, logger.Logger | None]":
    """
    Setup prerequisites for logging in `main()` .

    config: The configuration.

    Returns: Success, logger.
    """
    # Get settings
    try:
        log_directory_path = config["logger"]["directory_path"]
        log_path_format = config["logger"]["file_datetime_format"]
        start_time = datetime.datetime.now().strftime(log_path_format)
    except KeyError as exception:
        print(f"ERROR: Config key(s) not found: {exception}")
        return False, None

    # Create logging directory
    pathlib.Path(log_directory_path).mkdir(exist_ok=True)
    pathlib.Path(log_directory_path, start_time).mkdir()

    # Setup logger
    result, main_logger = logger.Logger.create(MAIN_LOGGER_NAME, True)
    if not result:
        print("ERROR: Failed to create main logger")
        return False, None

    assert main_logger is not None

    frame = inspect.currentframe()
    main_logger.info(f"{MAIN_LOGGER_NAME} logger initialized", frame)

    return True, main_logger
