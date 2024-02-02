import pytest
from modules.decision import Decision
from modules import decision_command
from modules import object_in_world
from modules import odometry_and_time
from modules import drone_odometry_local

# Test parameters
TOLERANCE = 2


@pytest.fixture()
def decision_maker():
    """
    Construct a Decision instance with predefined tolerance.
    """
    decision_instance = Decision(TOLERANCE)
    yield decision_instance


# Fixture for a pad within tolerance
@pytest.fixture()
def best_pad_within_tolerance():
    """
    Create a mock ObjectInWorld instance within tolerance.
    """
    position_x = 10.0
    position_y = 20.0
    spherical_variance = 1.0
    success, pad = object_in_world.ObjectInWorld.create(
        position_x, position_y, spherical_variance
    )
    assert success
    return pad


# Fixture for a pad outside tolerance
@pytest.fixture()
def best_pad_outside_tolerance():
    """
    Create a mock ObjectInWorld instance outside tolerance.
    """
    position_x = 100.0
    position_y = 200.0
    spherical_variance = 5.0  # variance outside tolerance
    success, pad = object_in_world.ObjectInWorld.create(
        position_x, position_y, spherical_variance
    )
    assert success
    return pad


# Fixture for a list of pads
@pytest.fixture()
def pads():
    """
    Create a list of mock ObjectInWorld instances.
    """
    pad1 = object_in_world.ObjectInWorld.create(30.0, 40.0, 2.0)[1]
    pad2 = object_in_world.ObjectInWorld.create(50.0, 60.0, 3.0)[1]
    pad3 = object_in_world.ObjectInWorld.create(70.0, 80.0, 4.0)[1]
    return [pad1, pad2, pad3]


# Fixture for odometry and time states
@pytest.fixture()
def states():
    """
    Create a mock OdometryAndTime instance with the drone positioned within tolerance of the landing pad.
    """
    # Creating the position within tolerance of the specified landing pad.
    position = drone_odometry_local.DronePositionLocal.create(9.0, 19.0, -5.0)[
        1
    ]  # Example altitude of -5 meters

    orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)[
        1
    ]

    odometry_data = drone_odometry_local.DroneOdometryLocal.create(
        position, orientation
    )[1]

    # Creating the OdometryAndTime instance with current time stamp
    success, state = odometry_and_time.OdometryAndTime.create(odometry_data)
    assert success
    return state


class TestDecision:
    """
    Tests for the Decision.run() method.
    """

    def test_decision_within_tolerance(
        self, decision_maker, best_pad_within_tolerance, pads, states
    ):
        """
        Test decision making when the best pad is within tolerance.
        """
        total_pads = [best_pad_within_tolerance] + pads
        command = decision_maker.run(states, total_pads)

        assert isinstance(command, decision_command.DecisionCommand)
        assert (
            command.get_command_type()
            == decision_command.DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION
        )

    def test_decision_outside_tolerance(
        self, decision_maker, best_pad_outside_tolerance, pads, states
    ):
        """
        Test decision making when the best pad is outside tolerance.
        """
        total_pads = [best_pad_outside_tolerance] + pads
        command = decision_maker.run(states, total_pads)

        assert isinstance(command, decision_command.DecisionCommand)
        assert (
            command.get_command_type()
            == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION
        )

    def test_decision_no_pads(self, decision_maker, states):
        """
        Test decision making when no pads are available.
        """
        command = decision_maker.run(states, [])

        assert isinstance(command, decision_command.DecisionCommand)
        assert (
            command.get_command_type()
            == decision_command.DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION
        )
