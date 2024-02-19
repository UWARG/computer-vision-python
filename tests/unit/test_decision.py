"""
Tests the decision class.
"""

import pytest

from modules.decision import decision
from modules import decision_command
from modules import object_in_world
from modules import odometry_and_time
from modules import drone_odometry_local


LANDING_PAD_LOCATION_TOLERANCE = 2

BEST_PAD_LOCATION_X = 10.0
BEST_PAD_LOCATION_Y = 20.0

DRONE_OFFSET_FROM_PAD = 1.0


# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture()
def decision_maker():
    """
    Construct a Decision instance with predefined tolerance.
    """
    decision_instance = decision.Decision(LANDING_PAD_LOCATION_TOLERANCE)
    yield decision_instance


@pytest.fixture()
def best_pad_within_tolerance():
    """
    Create an ObjectInWorld instance within distance to pad tolerance.
    """
    location_x = BEST_PAD_LOCATION_X
    location_y = BEST_PAD_LOCATION_Y
    spherical_variance = 1.0
    result, pad = object_in_world.ObjectInWorld.create(location_x, location_y, spherical_variance)
    assert result
    assert pad is not None

    yield pad


@pytest.fixture()
def best_pad_outside_tolerance():
    """
    Creates an ObjectInWorld instance outside of distance to pad tolerance.
    """
    location_x = 100.0
    location_y = 200.0
    spherical_variance = 5.0  # variance outside tolerance
    result, pad = object_in_world.ObjectInWorld.create(location_x, location_y, spherical_variance)
    assert result
    assert pad is not None

    yield pad


@pytest.fixture()
def pads():
    """
    Create a list of ObjectInWorld instances for the landing pads.
    """
    result, pad1 = object_in_world.ObjectInWorld.create(30.0, 40.0, 2.0)
    assert result
    assert pad1 is not None

    result, pad2 = object_in_world.ObjectInWorld.create(50.0, 60.0, 3.0)
    assert result
    assert pad2 is not None

    result, pad3 = object_in_world.ObjectInWorld.create(70.0, 80.0, 4.0)
    assert result
    assert pad3 is not None

    yield [pad1, pad2, pad3]


@pytest.fixture()
def drone_odometry_and_time():
    """
    Create an OdometryAndTime instance with the drone positioned within tolerance of landing pad.
    """
    # Creating the position within tolerance of the specified landing pad.
    result, position = drone_odometry_local.DronePositionLocal.create(
        BEST_PAD_LOCATION_X - DRONE_OFFSET_FROM_PAD,
        BEST_PAD_LOCATION_Y - DRONE_OFFSET_FROM_PAD,
        -5.0,
    )
    assert result
    assert position is not None

    result, orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)
    assert result
    assert orientation is not None

    result, odometry_data = drone_odometry_local.DroneOdometryLocal.create(position, orientation)
    assert result
    assert odometry_data is not None

    # Creating the OdometryAndTime instance with current time stamp
    result, odometry_with_time = odometry_and_time.OdometryAndTime.create(odometry_data)
    assert result
    assert odometry_with_time is not None

    yield odometry_with_time


class TestDecision:
    """
    Tests for the Decision.run() method and weight and distance methods.
    """

    def test_decision_within_tolerance(
        self, decision_maker, best_pad_within_tolerance, pads, drone_odometry_and_time
    ):
        """
        Test decision making when the best pad is within tolerance.
        """
        expected = decision_command.DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION
        total_pads = [best_pad_within_tolerance] + pads

        result, command = decision_maker.run(drone_odometry_and_time, total_pads)

        assert result
        assert command is not None
        assert command.get_command_type() == expected

    def test_decision_outside_tolerance(
        self, decision_maker, best_pad_outside_tolerance, pads, drone_odometry_and_time
    ):
        """
        Test decision making when the best pad is outside tolerance.
        """
        expected = decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION
        total_pads = [best_pad_outside_tolerance] + pads

        result, command = decision_maker.run(drone_odometry_and_time, total_pads)

        assert result
        assert command is not None
        assert command.get_command_type() == expected

    def test_decision_no_pads(self, decision_maker, drone_odometry_and_time):
        """
        Test decision making when no pads are available.
        """
        result, command = decision_maker.run(drone_odometry_and_time, [])

        assert not result
        assert command is None
