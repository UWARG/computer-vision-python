from .. import decision_command
from ..cluster_estimation import cluster_estimation
from ..flight_interface import flight_interface

class Decision:

    def __init__(self):
        self.__best_landing_pad = 0
        self.__res1, self.__landing_pad_states = cluster_estimation.ClusterEstimation.run()
        self.__res2, self.__current_pos = flight_interface.FlightInterface.run()
        self.__weighted_pads = []

    def distance_to_pad(self, pad):
        """
        finds distance to landing pad based of current position
        """

    def weight_pads(self):
        """
        weights the pads based on variance and distance
        """
    def __find_best_pad(self):
        """
        uses list of tuple(pad, weight) to determine best pad
        """

    def run(self,
            states, variance) -> decision_command.DecisionCommand:
        return decision_command.DecisionCommand.CommandType.LAND_AT_CURRENT_POSITION
