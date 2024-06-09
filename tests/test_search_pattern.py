"""
Tests the SearchPattern class
"""

from typing import Generator
import pytest

from modules import decision_command
from modules import odometry_and_time
from modules import drone_odometry_local
from modules.decision import search_pattern

DISTANCE_SQUARED_THRESHOLD = 2.0
SMALL_ADJUSTMENT = 1

# Test functions use test fixture signature names and access class privates
# pylint: disable=redefined-outer-name


@pytest.fixture()
def search_pattern_width_greater_depth() -> Generator[search_pattern.SearchPattern, None, None]:
    """
    Initializes a SearchPattern instance.
    """
    search_pattern_instance = search_pattern.SearchPattern(
        camera_fov_forwards=30.0,
        camera_fov_sideways=60.0,
        search_height=100.0,
        search_overlap=0.2,
        current_position_x=100.0,
        current_position_y=50.0,
        distance_squared_threshold=DISTANCE_SQUARED_THRESHOLD,
        small_adjustment=SMALL_ADJUSTMENT,
    )
    yield search_pattern_instance


@pytest.fixture()
def search_pattern_depth_greater_width() -> Generator[search_pattern.SearchPattern, None, None]:
    """
    Initializes a SearchPattern instance.
    """
    search_pattern_instance = search_pattern.SearchPattern(
        camera_fov_forwards=60.0,
        camera_fov_sideways=30.0,
        search_height=50.0,
        search_overlap=0.2,
        current_position_x=75.0,
        current_position_y=150.0,
        distance_squared_threshold=DISTANCE_SQUARED_THRESHOLD,
        small_adjustment=SMALL_ADJUSTMENT,
    )
    yield search_pattern_instance


@pytest.fixture()
def drone_odometry() -> Generator[odometry_and_time.OdometryAndTime, None, None]:
    """
    Pytest Fixture for coordinates of drone
    """

    def create_drone_odometry(
        pos_x: float, pos_y: float, depth: float
    ) -> odometry_and_time.OdometryAndTime:
        """
        Creates an OdometryAndTime instance representing the drone's current position.
        """
        position = drone_odometry_local.DronePositionLocal.create(pos_y, pos_x, -depth)[1]
        orientation = drone_odometry_local.DroneOrientationLocal.create_new(0.0, 0.0, 0.0)[1]
        odometry_data = drone_odometry_local.DroneOdometryLocal.create(position, orientation)[1]

        odometry = odometry_and_time.OdometryAndTime.create(odometry_data)[1]
        return odometry

    return create_drone_odometry


def assert_move_command(
    move: decision_command.DecisionCommand,
    expected_new_pos: bool,
    expected_target_x: float,
    expected_target_y: float,
    expected_target_z: float,
) -> None:
    """
    Compares the passed in move command with the expected values for this command
    """
    expected_command_type = decision_command.DecisionCommand.CommandType.MOVE_TO_ABSOLUTE_POSITION

    actual_new_pos = move[0]
    actual_target_pos = move[1].get_command_position()
    actual_targetx = actual_target_pos[0]
    actual_targety = actual_target_pos[1]
    actual_targetz = actual_target_pos[2]
    actual_command_type = move[1].get_command_type()

    assert actual_new_pos == expected_new_pos
    assert actual_targetx == pytest.approx(expected_target_x, 0.1)
    assert actual_targety == pytest.approx(expected_target_y, 0.1)
    assert actual_targetz == pytest.approx(expected_target_z, 0.1)
    assert actual_command_type == expected_command_type


class TestSearchPattern:
    """
    Tests for the SearchPattern class methods
    """

    def test_width_greater_depth_normal(
        self,
        search_pattern_width_greater_depth: search_pattern.SearchPattern,
        drone_odometry: odometry_and_time.OdometryAndTime,
    ) -> None:
        """
        Test first 20 positions of search pattern where drone has reached point before being called
        """
        expected_coordinates = [
            [100, 50, True],
            [123.75, 92.87, False],
            [124.75, 92.87, True],
            [142.87, 118.62, False],
            [142.87, 117.62, True],
            [142.87, 74.75, True],
            [168.62, 7.13, False],
            [167.62, 7.13, True],
            [124.75, 7.13, True],
            [57.13, -18.62, False],
            [57.13, -17.62, True],
            [57.13, 18.00, True],
            [57.13, 53.62, True],
            [57.13, 89.25, True],
            [57.13, 124.87, True],
            [31.38, 185.25, False],
            [32.38, 185.25, True],
            [68.00, 185.25, True],
            [103.62, 185.25, True],
            [139.25, 185.25, True],
            [174.87, 185.25, True],
        ]

        for i in range(len(expected_coordinates) - 1):
            current_position = drone_odometry(
                expected_coordinates[i][0], expected_coordinates[i][1], 100
            )
            move_command = search_pattern_width_greater_depth.continue_search(current_position)
            assert_move_command(
                move_command,
                expected_coordinates[i + 1][2],
                expected_coordinates[i + 1][0],
                expected_coordinates[i + 1][1],
                -100,
            )

    def test_depth_greater_width_normal(
        self,
        search_pattern_depth_greater_width: search_pattern.SearchPattern,
        drone_odometry: odometry_and_time.OdometryAndTime,
    ) -> None:
        """
        Test first 20 positions of search pattern where drone has reached point before being called
        """
        expected_coordinates = [
            [75, 150, True],
            [74.00, 171.44, False],
            [75.00, 171.44, True],
            [96.44, 172.44, False],
            [96.44, 171.44, True],
            [96.44, 143.81, True],
            [97.44, 128.56, False],
            [96.44, 128.56, True],
            [68.81, 128.56, True],
            [53.56, 127.56, False],
            [53.56, 128.56, True],
            [53.56, 166.91, True],
            [52.56, 192.87, False],
            [53.56, 192.87, True],
            [91.91, 192.87, True],
            [117.87, 193.87, False],
            [117.87, 192.87, True],
            [117.87, 160.17, True],
            [117.87, 127.46, True],
            [118.87, 107.13, False],
            [117.87, 107.13, True],
        ]

        for i in range(len(expected_coordinates) - 1):
            current_position = drone_odometry(
                expected_coordinates[i][0], expected_coordinates[i][1], 50
            )
            move_command = search_pattern_depth_greater_width.continue_search(current_position)
            assert_move_command(
                move_command,
                expected_coordinates[i + 1][2],
                expected_coordinates[i + 1][0],
                expected_coordinates[i + 1][1],
                -50,
            )

    def test_return_to_pattern(
        self,
        search_pattern_width_greater_depth: search_pattern.SearchPattern,
        drone_odometry: odometry_and_time.OdometryAndTime,
    ) -> None:
        """
        Test behaviour when drone has not reached the next waypoint
        """
        current_position = drone_odometry(100, 50, 100)
        move_command = search_pattern_width_greater_depth.continue_search(current_position)
        assert_move_command(move_command, False, 125.75, 92.87, -100)

        # Pretend drone goes off to search
        current_position = drone_odometry(200, 150, 100)
        move_command = search_pattern_width_greater_depth.continue_search(current_position)
        # Should still have same target location
        assert_move_command(move_command, False, 125.75, 92.87, -100)
