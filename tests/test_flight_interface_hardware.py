"""
Simple hardware test, requires drone connection.
"""

from modules.flight_interface import flight_interface


MAVLINK_CONNECTION_ADDRESS = "tcp:localhost:14550"
TIMEOUT_HOME = 10.0  # seconds


if __name__ == "__main__":
    # Setup
    result, interface = flight_interface.FlightInterface.create(
        MAVLINK_CONNECTION_ADDRESS,
        TIMEOUT_HOME,
    )
    assert result
    assert interface is not None

    # Run
    result, odometry_time = interface.run()

    # Test
    assert result
    assert odometry_time is not None

    print("Done!")
