"""
Creates decision for next action based on current state and detected pads
"""

from .. import decision_command
from .. import object_in_world
from .. import odometry_and_time


class Decision:
    """
    Weighs distance to located pads and variance to choose the next action, either to land or move to the nearest pad
    """

    def __init__(self, tolerance: float):
        self.__best_landing_pad = None
        self.__weighted_pads = []
        self.__distance_tolerance = tolerance

    @staticmethod
    def distance_to_pad(
        pad: object_in_world.ObjectInWorld,
        current_position: odometry_and_time.OdometryAndTime,
    ):
        """
        Calculate Euclidean distance to landing pad based on current position.
        """
        dx = pad.position_x - current_position.odometry_data.position.north
        dy = pad.position_y - current_position.odometry_data.position.east
        return (dx**2 + dy**2) ** 0.5

    def __weight_pads(
        self,
        pads: "list[object_in_world.ObjectInWorld]",
        current_position: odometry_and_time.OdometryAndTime,
    ):
        """
        Weights the pads based on normalized variance and distance.
        """
        if not pads:
            return None

        distances = [self.distance_to_pad(pad, current_position) for pad in pads]
        variances = [pad.spherical_variance for pad in pads]

        max_distance = max(distances)

        max_variance = max(variances)

        # if max distance is 0, assumes target pad is directly below, should land
        if max_distance == 0:
            self.__weighted_pads = [(pads[0], 0)]
            return None

        # if all variance is 0, no pads are found
        if max_variance == 0:
            return None

        self.__weighted_pads = [
            (pad, distance / max_distance + variance / max_variance)
            for pad, distance, variance in zip(pads, distances, variances)
        ]

    def __find_best_pad(self):
        """
        Determine the best pad to land on based on the weighted scores.
        """
        if not self.__weighted_pads:
            return None
        # Find the pad with the smallest weight as the best pad
        self.__best_landing_pad = min(self.__weighted_pads, key=lambda x: x[1])[0]
        return self.__best_landing_pad

    def run(
        self,
        curr_state: odometry_and_time.OdometryAndTime,
        pads: "list[object_in_world.ObjectInWorld]",
    ):
        """
        Determine the best landing pad and issue a command to land there.
        """
        self.__weight_pads(pads, curr_state)
        best_pad = self.__find_best_pad()
        if best_pad:
            distance_to_best_bad = self.distance_to_pad(best_pad, curr_state)

            # Issue a landing command if within tolerance
            if distance_to_best_bad <= self.__distance_tolerance:
                return (
                    True,
                    decision_command.DecisionCommand.create_land_at_absolute_position_command(
                        best_pad.position_x,
                        best_pad.position_y,
                        curr_state.odometry_data.position.down,
                    ),
                )
            # Move to best location if not within tolerance
            else:
                return (
                    True,
                    decision_command.DecisionCommand.create_move_to_absolute_position_command(
                        best_pad.position_x,
                        best_pad.position_y,
                        -curr_state.odometry_data.position.down,  # Assuming down is negative for landing
                    ),
                )
        # Default to do nothing if no pads are found
        else:
            return False, None
