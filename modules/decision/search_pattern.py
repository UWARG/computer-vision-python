from .. import decision_command
from .. import object_in_world
from .. import odometry_and_time
from math import tan, pi, cos, sin

"""

"""

class SearchPattern:
    def __init__(self,
                 camera_fov: float,#fov should be the smallest measurement (e.g. vertical for landscape) essentially assumes its a cricle with that angle
                 search_height: float,
                 search_overlap: float,
                 current_position: odometry_and_time.OdometryAndTime):#search overlap is float between 0 and 1. Represents how much overlap between passes (1 means fully overlaps (i.e. do not use). Likely value 0.2-0.5)
        self.current_ring = 0
        self.current_pos_in_ring = 0
        self.max_pos_in_ring = 0
        self.search_radius = 0

        self.acceptable_variance_squared = 1 #CHANGE THIS VALUE
        
        self.search_start_pos = current_position
        self.target_posx = self.search_start_pos.odometry_data.position.north
        self.target_posy = self.search_start_pos.odometry_data.position.east

        self.search_width = 2 * search_height * tan(camera_fov / 2)
        self.search_gap = self.search_width * (1 - search_overlap)


    def set_next_location(self):
        if self.current_pos_in_ring >= self.max_pos_in_ring:
            self.current_ring += 1
            self.current_pos_in_ring = 0
            self.search_radius = self.search_gap * self.current_ring
            self.max_pos_in_ring = self.search_radius * 2 * pi / self.search_gap
        else:
            self.current_pos_in_ring += 1

    
    def find_current_location(self) -> object_in_world.ObjectInWorld:
        # Based on curent ring and current pos in ring, set target_posx and target_posy
        self.relative_target_posx = self.search_radius * cos(2 * pi * self.current_pos_in_ring / self.max_pos_in_ring)
        self.relative_target_posy = self.search_radius * sin(2 * pi * self.current_pos_in_ring / self.max_pos_in_ring)
        self.target_posx = self.search_start_pos.odometry_data.position.north + self.relative_target_posx
        self.target_posy = self.search_start_pos.odometry_data.position.east + self.relative_target_posy
        return
    
    def distance_to_target_squared(self,
                                   current_position: odometry_and_time.OdometryAndTime) -> float:
        return (self.target_posx - current_position.odometry_data.position.east) ** 2 + (self.target_posy - current_position.odometry_data.position.north) ** 2

    def continue_search(self, current_position: odometry_and_time.OdometryAndTime) -> decision_command.DecisionCommand:
        #if drone is at target position, set next location, otherwise, move to target location
        if self.distance_to_target_squared(current_position) < self.acceptable_variance_squared:
            self.set_next_location()
            self.find_current_location()

        return decision_command.DecisionCommand.create_move_to_absolute_position_command(self.target_posx, self.target_posy, current_position.odometry_data.position.down)
    
