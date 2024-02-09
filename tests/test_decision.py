"""
Tests the decision class
"""

import pytest


from modules.decision import decision
from modules import decision_command
from modules import object_in_world
from modules import odometry_and_time
from modules import drone_odometry_local


LANDING_PAD_LOCATION_TOLERANCE = 2  # Test parameters


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
    Create a mock ObjectInWorld instance within tolerance.
    """
    position_x = 10.0
    position_y = 20.0
    spherical_variance = 1.0
    success, pad = object_in_world.ObjectInWorld.create(
        position_x, position_y, spherical_variance
    )
    assert success
    yield pad


@pytest.fixture()
def best_pad_outside_tolerance():
    """
    Creates an ObjectInWorld instance outside of distance to pad tolerance.
    """
    position_x = 100.0
    position_y = 200.0
    spherical_variance = 5.0  # variance outside tolerance
    result, pad = object_in_world.ObjectInWorld.create(
        position_x, position_y, spherical_variance
    )
    assert result
    yield pad


@pytest.fixture()
def pads():
    """
    Create a list of mock ObjectInWorld instances.
    """
    pad1 = object_in_world.ObjectInWorld.create(30.0, 40.0, 2.0)[1]
    pad2 = object_in_world.ObjectInWorld.create(50.0, 60.0, 3.0)[1]
    pad3 = object_in_world.ObjectInWorld.create(70.0, 80.0, 4.0)[1]
    yield [pad1, pad2, pad3]


@pytest.fixture()
def state():
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
    result, state = odometry_and_time.OdometryAndTime.create(odometry_data)
    assert result
    yield state


class TestDecision:
    """
    Tests for the Decision.run() method and weight and distance methods
    """        

    def test_decision_within_tolerance(self, 
                                       decision_maker, 
                                       best_pad_within_tolerance, 
                                       pads, 
                                       state):
        """
        Test decision making when the best pad is within tolerance.
        """
        expected = decision_command.DecisionCommand.CommandType.LAND_AT_ABSOLUTE_POSITION
        total_pads = [best_pad_within_tolerance] + pads
        
        result, actual = decision_maker.run(state, total_pads)

        assert result
        assert actual.get_command_type() == expected

    def test_decision_outside_tolerance(self, 
                                        decision_maker, 
                                        best_pad_outside_tolerance, 
                                        pads, 
                                        state):
        """
        Test decision making when the best pad is outside tolerance.
        """
        expected = decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION
        total_pads = [best_pad_outside_tolerance] + pads
        
        result, actual = decision_maker.run(state, total_pads)

        assert result
        assert actual.get_command_type() == expected

    def test_decision_no_pads(self, 
                              decision_maker, 
                              state):
        """
        Test decision making when no pads are available.
        """
        expected = None
        
        result, actual = decision_maker.run(state, [])

        assert result == False
        assert actual == expected
        