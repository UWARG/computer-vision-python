"""
Creates decision for next action based on current state and detected pads.
"""

from .. import decision_command
from .. import object_in_world
from .. import odometry_and_time


class ScoredLandingPad:
    """
    Landing pad with score for decision.
    """

    def __init__(self, landing_pad: object_in_world.ObjectInWorld, score: float) -> None:
        self.landing_pad = landing_pad
        self.score = score


class Decision:
    """
    Chooses next action to take based on known landing pad information.
    """

    def __init__(self, tolerance: float) -> None:
        self.__best_landing_pad: "object_in_world.ObjectInWorld | None" = None
        self.__weighted_pads = []
        self.__distance_tolerance = tolerance

    @staticmethod
    def __distance_to_pad(
        pad: object_in_world.ObjectInWorld, current_position: odometry_and_time.OdometryAndTime
    ) -> float:
        """
        Calculate squared Euclidean distance to landing pad based on current position.
        """
        dx = pad.location_x - current_position.odometry_data.position.north
        dy = pad.location_y - current_position.odometry_data.position.east
        return dx**2 + dy**2

    def __weight_pads(
        self,
        pads: "list[object_in_world.ObjectInWorld]",
        current_position: odometry_and_time.OdometryAndTime,
    ) -> "list[ScoredLandingPad] | None":
        """
        Weights the pads based on normalized variance and distance.
        """
        if len(pads) == 0:
            return None

        distances = [self.__distance_to_pad(pad, current_position) for pad in pads]
        variances = [pad.spherical_variance for pad in pads]

        max_distance = max(distances)

        max_variance = max(variances)

        # if all variance is zero, assumes no significant difference amongst pads
        # if max_distance is zero, assumes landing pad is directly below
        if max_variance == 0 or max_distance == 0:
            return [ScoredLandingPad(pad, 0) for pad in pads]

        return [
            ScoredLandingPad(pad, distance / max_distance + variance / max_variance)
            for pad, distance, variance in zip(pads, distances, variances)
        ]

    @staticmethod
    def __find_best_pad(
        weighted_pads: "list[ScoredLandingPad]",
    ) -> "object_in_world.ObjectInWorld | None":
        """
        Determine the best pad to land on based on the weighted scores.
        """
        if len(weighted_pads) == 0:
            return None

        # Find the pad with the smallest weight as the best pad
        best_landing_pad = min(weighted_pads, key=lambda pad: pad.score).landing_pad
        return best_landing_pad

    def run(
        self,
        curr_state: odometry_and_time.OdometryAndTime,
        pads: "list[object_in_world.ObjectInWorld]",
    ) -> "tuple[bool, decision_command.DecisionCommand | None]":
        """
        Determine the best landing pad and issue a command to land there.
        """
        self.__weighted_pads = self.__weight_pads(pads, curr_state)

        if self.__weighted_pads is None:
            return False, None

        self.__best_landing_pad = self.__find_best_pad(self.__weighted_pads)
        if self.__best_landing_pad is None:
            return False, None

        # Get Pylance to stop complaining
        assert self.__best_landing_pad is not None

        distance_to_best_pad = self.__distance_to_pad(self.__best_landing_pad, curr_state)

        # Issue a landing command if over the landing pad
        if distance_to_best_pad <= self.__distance_tolerance:
            return (
                True,
                decision_command.DecisionCommand.create_land_at_absolute_position_command(
                    self.__best_landing_pad.location_x,
                    self.__best_landing_pad.location_y,
                    curr_state.odometry_data.position.down,
                ),
            )

        # Move to landing pad
        return (
            True,
            decision_command.DecisionCommand.create_move_to_absolute_position_command(
                self.__best_landing_pad.location_x,
                self.__best_landing_pad.location_y,
                curr_state.odometry_data.position.down,
            ),
        )
