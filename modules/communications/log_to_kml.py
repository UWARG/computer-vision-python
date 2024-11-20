"""
Convert log file to KML file.
"""

import pathlib
import re

from modules.common.modules.kml.kml_conversion import locations_to_kml
from modules.common.modules.location_global import LocationGlobal


def convert_log_to_kml(
    log_file: str, document_name_prefix: str, save_directory: str
) -> "tuple[bool, pathlib.Path | None]":
    """Given a log file with a specific format, return a corresponding KML file.

    Args:
        log_file (str): Path to the log file
        document_name_prefix (str): Prefix name for saved KML file.
        save_directory (str): Directory to save the KML file to.

    Returns:
        tuple[bool, pathlib.Path | None]: Returns (False, None) if function
            failed to execute, otherwise (True, path) where path a pathlib.Path
            object pointing to the KML file.
    """
    locations = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                # find all the latitudes and longitudes within the line
                latitudes = re.findall(r"latitude: (-?\d+\.\d+)", line)
                longitudes = re.findall(r"longitude: (-?\d+\.\d+)", line)

                # we must find equal number of latitude and longitude numbers,
                # otherwise that means the log file is improperly formatted or
                # the script failed to detect all locations
                if len(latitudes) != len(longitudes):
                    print("Number of latitudes and longitudes found are different.")
                    print(f"# of altitudes: {len(latitudes)}, # of longitudes: {len(longitudes)}")
                    return False, None

                latitudes = list(map(float, latitudes))
                longitudes = list(map(float, longitudes))

                for i in range(len(latitudes)):
                    success, location = LocationGlobal.create(latitudes[i], longitudes[i])
                    if not success:
                        return False, None
                    locations.append(location)

            return locations_to_kml(locations, document_name_prefix, save_directory)
    except Exception as e:
        print(e.with_traceback())
        return False, None
