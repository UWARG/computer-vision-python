from .. import decision_command
from .. import odometry_and_time
from math import tan, pi, cos, sin, ceil

class SearchPattern:
    """
    Implements a search pattern based on camera field of view, search height, and overlap.
    
    Attributes:
        camera_fov (float): 
            Camera's field of view, measured in degrees. Use the smallest measurement available.
            Is essentially assumed to be a circle
        search_height (float): 
            The altitude at which the search is conducted. This is used to calculate the pattern.
            It does not make the drone go to this height
        search_overlap (float): 
            Overlap between passes, between 0 and 1. Recomended probably 0.5-0.7.
        current_position (OdometryAndTime):
            The drone's current position.
        acceptable_variance_squared (float):
            The square of the acceptable distance from the target position.
    """

    def __init__(self,
                 camera_fov: float,
                 search_height: float,
                 search_overlap: float,
                 current_position: odometry_and_time.OdometryAndTime,
                 acceptable_variance_squared: float):
        
        # Initialize the drone to the first position in the search pattern
        self.current_ring = 0
        self.current_pos_in_ring = 0
        self.max_pos_in_ring = 0
        self.search_radius = 0
        self.acceptable_variance_squared = acceptable_variance_squared
        
        # Store the origin of the search and 
        self.search_origin = current_position.odometry_data.position
        self.target_posx = self.search_origin.north
        self.target_posy = self.search_origin.east

        # Calculate the gap between positions
        self.search_width = 2 * search_height * tan((camera_fov * 180 / pi) / 2)
        self.search_gap = self.search_width * (1 - search_overlap)
    
    def set_target_location(self):
        """
        Updates the target position to the next position in the search pattern.
        """

        # If it is done the current ring, move to the next ring. If not, next step in current ring
        if self.current_pos_in_ring >= self.max_pos_in_ring:
            self.current_ring += 1
            self.current_pos_in_ring = 0
            self.search_radius = self.search_gap * self.current_ring
            self.max_pos_in_ring = (ceil(self.search_radius * 2 * pi / self.search_gap))
            self.angle_between_positions = ((2 * pi) / (self.max_pos_in_ring + 1))
        else:
            self.current_pos_in_ring += 1

        # Angle measured counter-clockwise from x-axis for the target location
        self.angle = self.angle_between_positions * self.current_pos_in_ring

        # Calculate x and y coordinates of new target location
        self.relative_target_posx = self.search_radius * cos(self.angle)
        self.relative_target_posy = self.search_radius * sin(self.angle)

        self.target_posx = self.search_origin.north + self.relative_target_posx
        self.target_posy = self.search_origin.east + self.relative_target_posy
    
    def distance_to_target_squared(self,
                                   current_position: odometry_and_time.OdometryAndTime
                                   ) -> float:
        """
        Returns the square of the distance to it's target location
        """
        return ((self.target_posx - current_position.odometry_data.position.east) ** 2 
                + (self.target_posy - current_position.odometry_data.position.north) ** 2)

    def continue_search(self, 
                        current_position: odometry_and_time.OdometryAndTime
                        )-> decision_command.DecisionCommand:
        
        """
        Call this function to have the drone go to the next location in the search pattern
        """
        # If it is at it's current target location, update to the next target
        if self.distance_to_target_squared(current_position) < self.acceptable_variance_squared:
            self.set_target_location()

        # Send command to go to target
        return decision_command.DecisionCommand.create_move_to_absolute_position_command(
            self.target_posx,
            self.target_posy,
            current_position.odometry_data.position.down)
    