"""
Initiates/continues a search pattern when the drone cannot find a landing pad.
"""

from .. import decision_command
from .. import odometry_and_time
from math import tan, pi, ceil, copysign

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
            Overlap between passes, between 0 and 1. Recomended probably ~0.5, experiment with less.
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
                 current_position: odometry_and_time.OdometryAndTime,
                 distance_squared_threshold: float,
                 small_adjustment: float):
        
        # Initialize the drone to the first position in the search pattern
        self.current_square = 0
        self.current_side_in_square = 0
        self.current_pos_on_side = 0
        self.max_pos_on_side = 0

        self.distance_squared_threshold = distance_squared_threshold
        
        # Store the origin of the search
        self.small_adjustment = small_adjustment

        self.search_origin = current_position.odometry_data.position
        self.search_origin.north += small_adjustment
        self.target_posx = self.search_origin.north
        self.target_posy = self.search_origin.east


        # Calculate the gap between positions
        self.search_width = 2 * search_height * tan((camera_fov_sideways * 180 / pi) / 2)
        self.search_depth = 2 * search_height * tan((camera_fov_forwards * 180 / pi) / 2)
        self.search_gap_width = self.search_width * (1 - search_overlap)
        self.search_gap_depth = self.search_depth * (1 - search_overlap)

        self.square_corners = [(self.target_posx, self.target_posy)] * 4
    
    def calculate_square_corners(self):
        """
        Computes the 4 corners of the current square based on the specified logic to ensure each new
        square spirals outward from the previous one.
        """
        square_num = self.current_square + 1  # Increment the square number for each new square

        # Calculate base offsets for the current square
        offset_x = self.search_gap_width * square_num
        offset_y = self.search_gap_width * (square_num - 0.5) + self.search_depth * 0.5

        # Calculate the corners based on the offsets and the search origin
        self.square_corners = [
            (self.search_origin.east - offset_x, self.search_origin.north - offset_y),  # Top left corner
            (self.search_origin.east + offset_x, self.search_origin.north - offset_y),  # Top right corner
            (self.search_origin.east + offset_x, self.search_origin.north + offset_y),  # Bottom right corner
            (self.search_origin.east - offset_x, self.search_origin.north + offset_y),  # Bottom left corner
        ]

    def set_target_location(self):
        """
        Calculate and set the next target location for the drone to move to,
        considering that each side of the square might have a different length,
        and thus a different number of points where the drone needs to stop and scan.
        """
        # Calculate the current corner and the next corner to determine the current side's direction and length
        current_corner = self.square_corners[self.current_side_in_square]
        next_corner = self.square_corners[self.current_side_in_square + 1]

        # Determine if we are moving horizontally or vertically along the current side
        moving_horizontally = self.current_side_in_square % 2 == 0

        # Calculate the length of the current side based on the direction of movement
        if moving_horizontally:
            side_length = next_corner[0] - current_corner[0]
        else:
            side_length = next_corner[0] - current_corner[0]

        # Calculate the number of points (stops) needed along the current side, including the corner as a stopping point
        self.max_pos_on_side = ceil(abs(side_length) / self.search_gap_width)


        if self.current_pos_on_side < self.max_pos_on_side:
            # Calculate the fraction of the way to move towards the next corner
            fraction_along_side = self.current_pos_on_side / self.max_pos_on_side

            # Calculate the next target position based on the current fraction of the side covered
            dist_to_move = side_length * fraction_along_side + copysign((self.search_gap_width + self.search_gap_depth) / 2, side_length)
            if moving_horizontally:
                self.target_posx = current_corner[0] + dist_to_move
            else:
                self.target_posy = current_corner[1] + dist_to_move

            # Increment the position counter
            self.current_pos_on_side += 1
            if self.current_pos_on_side == 0:
                if moving_horizontally:
                    self.target_posx -= self.small_adjustment
                else:
                    self.target_posy -= self.small_adjustment
                return False
        else:
            # If we've reached the end of the current side, move to the next side
            self.current_pos_on_side = -1  # Reset position counter for the new side
            self.current_side_in_square = (self.current_side_in_square + 1) % 4  # Move to the next side
            if self.current_side_in_square == 0:
                # If we've circled back to the starting corner, it's time to move to a larger square
                self.current_square += 1
                self.calculate_square_corners()  # Recalculate corners for the new, larger square
            self.set_target_location()

        return True  # Indicate that a new target location has been set

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
