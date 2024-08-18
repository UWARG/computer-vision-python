"""
Simple hardware test, requires drone connection.
"""

import pathlib

from modules.flight_interface import flight_interface
from modules.common.logger.modules import logger


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
FLIGHT_INTERFACE_TIMEOUT = 10.0  # seconds
FLIGHT_INTERFACE_BAUD_RATE = 57600  # symbol rate


def main() -> int:
    """
    Main function.
    """
    # Logger
    test_name = pathlib.Path(__file__).stem
    result, local_logger = logger.Logger.create(test_name, False)
    assert result
    assert local_logger is not None

    # Setup
    result, interface = flight_interface.FlightInterface.create(
        MAVLINK_CONNECTION_ADDRESS,
        FLIGHT_INTERFACE_TIMEOUT,
        FLIGHT_INTERFACE_BAUD_RATE,
        local_logger,
    )
    assert result
    assert interface is not None

    # Run
    result, odometry_time = interface.run()

    # Test
    assert result
    assert odometry_time is not None

    return 0


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
