"""
Initiates/continues a search pattern when the drone cannot find a landing pad.
"""

from math import tan, pi, ceil
from enum import Enum
from .. import decision_command
from .. import odometry_and_time

# Class requires more than the default 7 allowed instance attributes
# pylint: disable=too-many-instance-attributes


class Corner(Enum):
    """
    Enum class used for square_corners object to acess the 4 corners
    """

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"


class Point:
    """
    2D cartesian points containing and x and y coordinate. Used in the square_corners object
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class SearchPattern:
    """
    Attributes:
        camera_fov_forwards (float):
            Camera's field of view, measured in degrees, in forwards/backwards direction
        camera_fov_sideways (float):
            Camera's field of view, measured in degrees, in left/right direction
        search_height (float):
            The altitude at which the search is conducted. This is used to calculate the pattern.
            It does not make the drone go to this height
        search_overlap (float):
            Overlap between passes, between 0 and 1.
        current_position (OdometryAndTime):
            The drone's current position.
        distance_squared_threshold (float):
            The square of the acceptable distance from the target position.
        small_adjustment (float):
            Small distance to ensure drone is facing the correct direction
    """

    @staticmethod
    def __distance_to_target_squared(
        current_position: odometry_and_time.OdometryAndTime,
        target_posx: float,
        target_posy: float,
    ) -> float:
        """
        Returns the square of the distance to it's target location
        """
        return (target_posx - current_position.odometry_data.position.east) ** 2 + (
            target_posy - current_position.odometry_data.position.north
        ) ** 2

    def __init__(
        self,
        camera_fov_forwards: float,
        camera_fov_sideways: float,
        search_height: float,
        search_overlap: float,
        current_position_x: float,
        current_position_y: float,
        distance_squared_threshold: float,
        small_adjustment: float,
    ) -> None:

        # Set local constants
        self.__distance_squared_threshold = distance_squared_threshold
        self.__small_adjustment = small_adjustment

        # The search will be centred around wherever it is initialized
        self.__search_origin_x = current_position_x
        self.__search_origin_y = current_position_y

        # Drone's current target
        self.__target_posx = self.__search_origin_x
        self.__target_posy = self.__search_origin_y

        # Initialize the drone to the first position in the search pattern
        self.__current_square = 1
        self.__current_side_in_square = Corner.TOP_LEFT
        self.__current_pos_on_side = -1

        # Calculate the gap between positions search_gap_width is the left/right distance
        # search_gap_depth is the forwards/backwards distance
        self.__search_gap_width = (
            2 * search_height * tan((camera_fov_sideways * pi / 180) / 2)
        ) * (1 - search_overlap)
        self.__search_gap_depth = (
            2 * search_height * tan((camera_fov_forwards * pi / 180) / 2)
        ) * (1 - search_overlap)

        # This variable is returned to signal whether the target location is one that should be
        # scanned, or just an intermediate point in its travel to the next
        self.__new_location = True

        # Calculate positions for first square
        self.__calculate_square_corners()
        self.__calculate_side_of_square()

    def __get_next_corner(self, corner: Corner) -> Corner:
        """
        Takes a corner and returns the next one in the square
        """
        if corner == Corner.TOP_LEFT:
            return Corner.TOP_RIGHT
        if corner == Corner.TOP_RIGHT:
            return Corner.BOTTOM_RIGHT
        if corner == Corner.BOTTOM_RIGHT:
            return Corner.BOTTOM_LEFT
        return Corner.TOP_LEFT

    def __calculate_square_corners(self) -> None:
        """
        Computes the 4 corners of the current square
        """
        # Compute the size of square. First square will be a distance of the minimum of width and
        # depth, each subsequent square will be a width larger
        square_size = (
            min(self.__search_gap_depth, self.__search_gap_width)
            + (self.__current_square - 1) * self.__search_gap_width
        )

        # If the depth is less than the width, we apply an offset at each corner to ensure the
        # entire area is scanned

        if self.__search_gap_depth < self.__search_gap_width:
            adjustment = (self.__search_gap_width - self.__search_gap_depth) / 2
        else:
            adjustment = 0

        # Calculate the corners based on the offsets and the search origin. Top left corner is moved
        # right by search_gap_width as the final side of the square will instead cover that part
        self.__square_corners = {
            Corner.TOP_LEFT: Point(
                self.__search_origin_x - square_size - adjustment + self.__search_gap_width,
                self.__search_origin_y + square_size,
            ),
            Corner.TOP_RIGHT: Point(
                self.__search_origin_x + square_size,
                self.__search_origin_y + square_size + adjustment,
            ),
            Corner.BOTTOM_RIGHT: Point(
                self.__search_origin_x + square_size + adjustment,
                self.__search_origin_y - square_size,
            ),
            Corner.BOTTOM_LEFT: Point(
                self.__search_origin_x - square_size,
                self.__search_origin_y - square_size - adjustment,
            ),
        }

    def __calculate_side_of_square(self) -> None:
        """
        Computes the gaps along the current side of the square
        """
        # Calculate the positions of the current and next corners
        self.__current_corner = self.__square_corners[self.__current_side_in_square]
        next_corner = self.__square_corners[self.__get_next_corner(self.__current_side_in_square)]

        # Determine if the drone is moving horizontally or vertically along the current side
        self.__moving_horizontally = self.__current_side_in_square in (
            Corner.TOP_LEFT,
            Corner.BOTTOM_RIGHT,
        )

        # Calculate the length of the current side based on the direction of movement
        # Note that this is signed (e.g. travellingin -x direction has a negative sign)
        if self.__moving_horizontally:
            side_length = next_corner.x - self.__current_corner.x
        else:
            side_length = next_corner.y - self.__current_corner.y

        # As calculated, it is the distance to the next corner, however, we want to stop before
        # we actually reach the next side as the area will be covered by it
        if side_length > 0:
            side_length += (self.__search_gap_depth - self.__search_gap_width) / 2
        else:
            side_length -= (self.__search_gap_depth - self.__search_gap_width) / 2

        # If we are on the last side of the square, we will continue until the next square to spiral
        if self.__current_side_in_square == Corner.BOTTOM_LEFT:
            side_length += self.__search_gap_width

        # Calculate the number of stops needed along the current side
        self.__max_pos_on_side = ceil(abs(side_length) / self.__search_gap_depth)

        self.__travel_gap = side_length / self.__max_pos_on_side

    def set_target_location(self) -> bool:
        """
        Calculate and set the next target location for the drone to move to.
        Returns true if the new target location is a spot we should scan at. If false, it means
        this point is an intermediate point so that the next one is facing the correct direction
        """
        # If we've reached the end of the current side, move to the next side
        if self.__current_pos_on_side >= self.__max_pos_on_side:
            self.__current_pos_on_side = -1  # Reset position counter for the new side
            self.__current_side_in_square = self.__get_next_corner(self.__current_side_in_square)

            if self.__current_side_in_square == Corner.TOP_LEFT:  # If completed this square
                self.__current_square += 1
                self.__calculate_square_corners()

            self.__calculate_side_of_square()

        # For the first position on a side, we set the drone a small amount off of the position so
        # that the next move turns the drone to face the correct direction
        if self.__current_pos_on_side == -1:
            if self.__current_side_in_square == Corner.TOP_LEFT:
                self.__target_posx = self.__current_corner.x - self.__small_adjustment
                self.__target_posy = self.__current_corner.y
            elif self.__current_side_in_square == Corner.TOP_RIGHT:
                self.__target_posx = self.__current_corner.x
                self.__target_posy = self.__current_corner.y + self.__small_adjustment
            elif self.__current_side_in_square == Corner.BOTTOM_RIGHT:
                self.__target_posx = self.__current_corner.x + self.__small_adjustment
                self.__target_posy = self.__current_corner.y
            elif self.__current_side_in_square == Corner.BOTTOM_LEFT:
                self.__target_posx = self.__current_corner.x
                self.__target_posy = self.__current_corner.y - self.__small_adjustment

            # Increment the position counter
            self.__current_pos_on_side += 1
            return False

        # Calculate the next target position based on the current fraction of the side covered
        if self.__current_pos_on_side == 0:
            self.__target_posx = self.__current_corner.x
            self.__target_posy = self.__current_corner.y
        elif self.__moving_horizontally:
            self.__target_posx += self.__travel_gap
        else:
            self.__target_posy += self.__travel_gap

        # Increment the position counter
        self.__current_pos_on_side += 1
        return True

    def continue_search(
        self, current_position: odometry_and_time.OdometryAndTime
    ) -> "tuple[bool, decision_command.DecisionCommand]":
        """
        Call this function to have the drone go to the next location in the search pattern
        The returned decisionCommand is the next target location, the boolean is if this new
        location is a point to scan or an intermediate point to travel to (i.e. call this function
        again after arriving at the destination to get the point to scan at)
        """

        # If it is at it's current target location, update to the next target.
        if (
            SearchPattern.__distance_to_target_squared(
                current_position, self.__target_posx, self.__target_posy
            )
            < self.__distance_squared_threshold
        ):
            self.__new_location = self.set_target_location()

        # Send command to go to target.
        return (
            self.__new_location,
            decision_command.DecisionCommand.create_move_to_absolute_position_command(
                self.__target_posx,
                self.__target_posy,
                current_position.odometry_data.position.down,
            ),
        )
