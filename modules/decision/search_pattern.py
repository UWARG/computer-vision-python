"""
Initiates/continues a search pattern when the drone cannot find a landing pad.
"""

from .. import decision_command
from .. import odometry_and_time
from math import tan, pi, ceil

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
    def __distance_to_target_squared(current_position: odometry_and_time.OdometryAndTime,
                                     target_posx: float,
                                     target_posy: float,
                                     ) -> float:
        """
        Returns the square of the distance to it's target location
        """
        return ((target_posx - current_position.odometry_data.position.east) ** 2
                + (target_posy - current_position.odometry_data.position.north) ** 2)

    def __init__(self,
                 camera_fov_forwards: float,
                 camera_fov_sideways: float,
                 search_height: float,
                 search_overlap: float,
                 current_position_x: float,
                 current_position_y: float,
                 distance_squared_threshold: float,
                 small_adjustment: float):

        # Store values to be used later
        self.distance_squared_threshold = distance_squared_threshold
        self.small_adjustment = small_adjustment

        # The search will be centred around wherever it is initialized
        self.search_origin_x = current_position_x
        self.search_origin_y = current_position_y

        # Drones current target
        self.target_posx = self.search_origin_x
        self.target_posy = self.search_origin_y

        # Initialize the drone to the first position in the search pattern
        self.current_square = 1
        self.current_side_in_square = 0
        self.current_pos_on_side = -1

        # Calculate the gap between positions search_gap_width is the left/right distance
        # search_gap_depth is the forwards/backwards distance
        self.search_width = 2 * search_height * tan((camera_fov_sideways * pi / 180) / 2)
        self.search_depth = 2 * search_height * tan((camera_fov_forwards * pi / 180) / 2)
        self.search_gap_width = self.search_width * (1 - search_overlap)
        self.search_gap_depth = self.search_depth * (1 - search_overlap)

        # Calculate positions for first square
        self.calculate_square_corners()
        self.calculate_side_of_square()

    def calculate_square_corners(self):
        """
        Computes the 4 corners of the current square
        """
        # Compute the size of square. First square will be a distance of the minimum of width and
        # depth, each subsequent square will be a width larger
        square_size = (min(self.search_gap_depth, self.search_gap_width)
                 + (self.current_square - 1) * self.search_gap_width)

        # If the depth is less than the width, we must adjust the sides of the square
        adjustment = 0
        if self.search_gap_depth < self.search_width:
            adjustment = (self.search_gap_width - self.search_gap_depth) / 2

        # Calculate the corners based on the offsets and the search origin
        #In order: Top left, top right, bottom right, bottom left.
        self.square_corners = [
            (self.search_origin_x - square_size - adjustment, self.search_origin_y + square_size),
            (self.search_origin_x + square_size, self.search_origin_y + square_size + adjustment),
            (self.search_origin_x + square_size + adjustment, self.search_origin_y - square_size),
            (self.search_origin_x - square_size, self.search_origin_y - square_size - adjustment),
        ]

    def calculate_side_of_square(self):
        """
        Computes the gaps along the current side of the square
        """
        # Calculate the current corner and the next corner
        self.current_corner = self.square_corners[self.current_side_in_square]
        next_corner = self.square_corners[(self.current_side_in_square + 1) % 4]

        # Determine if we are moving horizontally or vertically along the current side
        self.moving_horizontally = self.current_side_in_square % 2 == 0

        # Calculate the length of the current side based on the direction of movement
        if self.moving_horizontally:
            side_length = next_corner[0] - self.current_corner[0]
        else:
            side_length = next_corner[1] - self.current_corner[1]

        # Calculate the number of stops needed along the current side
        self.max_pos_on_side = ceil(abs(side_length) / self.search_gap_depth)

        self.travel_gap = side_length / self.max_pos_on_side

    def set_target_location(self):
        """
        Calculate and set the next target location for the drone to move to.
        Returns true if the new target location is a spot we should scan at. If false, it means
        this point is an intermediate point so that the next one is facing the correct direction
        """
        # If we've reached the end of the current side, move to the next side
        if self.current_pos_on_side >= self.max_pos_on_side:
            self.current_pos_on_side = -1  # Reset position counter for the new side
            self.current_side_in_square = (self.current_side_in_square + 1) % 4

            if self.current_side_in_square == 0: # If completed this square
                self.current_square += 1
                self.calculate_square_corners()
            self.calculate_side_of_square()

        # For the first position on a side, we set the drone a small amount off of the position so
        # that the next move turns the drone to face the correct direction
        if self.current_pos_on_side == -1:
            if self.current_side_in_square == 0:
                self.target_posx = self.current_corner[0] - self.small_adjustment
                self.target_posy = self.current_corner[1]
            elif self.current_side_in_square == 1:
                self.target_posx = self.current_corner[0]
                self.target_posy = self.current_corner[1] + self.small_adjustment
            elif self.current_side_in_square == 2:
                self.target_posx = self.current_corner[0] + self.small_adjustment
                self.target_posy = self.current_corner[1]
            elif self.current_side_in_square == 3:
                self.target_posx = self.current_corner[0]
                self.target_posy = self.current_corner[1] - self.small_adjustment

            self.current_pos_on_side += 1
            return False
        else:
            # Calculate the next target position based on the current fraction of the side covered
            dist_to_move = self.travel_gap * self.current_pos_on_side
            if self.moving_horizontally:
                self.target_posx = self.current_corner[0] + dist_to_move
            else:
                self.target_posy = self.current_corner[1] + dist_to_move

            # Increment the position counter
            self.current_pos_on_side += 1
            return True

    def continue_search(self,
                        current_position: odometry_and_time.OdometryAndTime
                        )-> "tuple[bool, decision_command.DecisionCommand]":

        """
        Call this function to have the drone go to the next location in the search pattern
        The returned decisionCommand is the next target location, the boolean is if this new
        location is a point to scan or an intermediate point to travel to (i.e. call this function
        again after arriving at the destination to get the point to scan at)
        """

        new_location = True

        # If it is at it's current target location, update to the next target.
        if (SearchPattern.__distance_to_target_squared(current_position,
                                                      self.target_posx,
                                                      self.target_posy)
                                                      < self.distance_squared_threshold):
            new_location = self.set_target_location()

        # Send command to go to target.
        return (new_location,
                decision_command.DecisionCommand.create_move_to_absolute_position_command(
                    self.target_posx,
                    self.target_posy,
                    current_position.odometry_data.position.down))
