"""
Simple hardware test, requires drone connection.
"""

import pytest

from modules.flight_interface import flight_interface


DRONE_CONNECTION = "tcp:localhost:14550"
TIMEOUT_HOME = 10.0  # seconds


@pytest.fixture
def interface():
    """
    Camera.
    """
    result, interfacer = flight_interface.FlightInterface.create(
        DRONE_CONNECTION,
        TIMEOUT_HOME,
    )
    assert result
    assert interfacer is not None

    yield interfacer


def test_video_input(interface: flight_interface.FlightInterface):
    """
    Test single image.
    """
    # Run
    result, odometry_time = interface.run()

    # Test
    assert result
    assert odometry_time is not None
