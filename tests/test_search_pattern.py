"""
Tests the SearchPattern class
"""

import pytest

from modules import decision_command
from modules import odometry_and_time
from modules import drone_odometry_local
from modules.decision import search_pattern


CAMERA_FOV = 90  # Camera's field of view in degrees
SEARCH_HEIGHT = 100  # Altitude at which the search is conducted
SEARCH_OVERLAP = 0.5  # Overlap between passes
DISTANCE_SQUARED_THRESHOLD = 1  # Acceptable variance squared


@pytest.fixture()
def drone_odometry():
    """
    Creates an OdometryAndTime instance representing the drone's current position.
    """
    position = drone_odometry_local.DronePositionLocal.create(0.0, 0.0, -SEARCH_HEIGHT)[1]
    orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)[1]
    odometry_data = drone_odometry_local.DroneOdometryLocal.create(position, orientation)[1]

    result, state = odometry_and_time.OdometryAndTime.create(odometry_data)
    assert result
    yield state


@pytest.fixture()
def search_maker(drone_odometry):
    """
    Initializes a SearchPattern instance.
    """
    search_pattern_instance = search_pattern.SearchPattern(
        camera_fov=CAMERA_FOV,
        search_height=SEARCH_HEIGHT,
        search_overlap=SEARCH_OVERLAP,
        current_position=drone_odometry,
        distance_squared_threshold=DISTANCE_SQUARED_THRESHOLD
    )
    yield search_pattern_instance


class TestSearchPattern:
    """
    Tests for the SearchPattern class methods
    """

    def test_initialization(self, search_maker):
        """
        Test the initialization of the SearchPattern object.
        """
        assert search_maker.search_radius == 0
        assert search_maker.current_ring == 0
        assert search_maker.current_pos_in_ring == 0
        assert search_maker.max_pos_in_ring == 0

    def test_continue_search_move_command(self, search_maker, drone_odometry):
        """
        Test continue_search method when drone is not at the target location.
        """
        search_maker.set_target_location() 
        command = search_maker.continue_search(drone_odometry)

        assert command.get_command_type() == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION

    def test_continue_search_no_move_needed(self, search_maker, drone_odometry):
        """
        Test continue_search method when the drone is already at the target location.
        """

        command = search_maker.continue_search(drone_odometry)

        assert command.get_command_type() == decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION

    def test_set_target_location_first_call(self, search_maker):
        """
        Test set_target_location method on its first call.
        """
        assert search_maker.current_pos_in_ring == 0  # First position in a ring is 0
        assert search_maker.current_ring == 0  # Should be in the 0th ring

        search_maker.set_target_location()

        assert search_maker.current_pos_in_ring == 0  # First position in a ring is 0
        assert search_maker.current_ring == 1  # Should switch to the first ring
