"""
For YAML files.
"""

import pathlib
import yaml


def open_config(file_path: pathlib.Path) -> "tuple[bool, dict | None]":
    """
    Open and decode YAML file.
    """
    try:
        with file_path.open("r", encoding="utf8") as file:
            try:
                config = yaml.safe_load(file)
                return True, config
            except yaml.YAMLError as exception:
                print(f"ERROR: Could not parse YAML file: {exception}")
    except FileNotFoundError as exception:
        print(f"ERROR: YAML file not found: {exception}")
    except IOError as exception:
        print(f"ERROR: Could not open file: {exception}")

    return False, None
