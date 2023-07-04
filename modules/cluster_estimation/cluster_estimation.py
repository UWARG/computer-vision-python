"""
TODO: Write description
"""

import sklearn
import numpy as np
# Parent directory import not working
from object_in_world import ObjectInWorld

# Placeholder:
from detection_in_world import DetectionInWorld

class ClusterEstimation:
    """
    TODO: Write description
    """
    def __init__(self, co_type="spherical", components=10, rand_state=0, weight_conc_prior=100, mean_precision_prior=1E-6,
                 min_points=100, new_points_per_run=10):
        # Initalizes VMGMM model 
        self.__vgmm_model = sklearn.mixture.BayesianGaussianMixture(
            covariance_type = co_type, 
            n_components = components, 
            random_state = rand_state, 
            weight_concentration_prior = weight_conc_prior, 
            init_params = 'k-means++', 
            mean_precision_prior = mean_precision_prior)
        # Points storage 
        self.__all_points = []
        self.__min_points = min_points
        self.__points_per_run = new_points_per_run
        self.__no_points_count = 0
        self.__current_bucket = []
        
    def run(self, detections: "list[DetectionInWorld]", run_override:  bool) -> "tuple[bool, list[ObjectInWorld | None]]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run
        """

        Z = self.__vgmm_model.predict(detections)
        print(Z)

        if not run_override and not self.decide_to_run(detections):
            return False, None

        # TODO: Implementation
        raise NotImplementedError

    def decide_to_run(self, detections: "list[DetectionInWorld]") -> bool:
        # Minimum detections to run
        min_to_run = 50
        if(len(detections) > min_to_run):
            return True
        else:
            return False


# Debug
simulated_detections, y = sklearn.datasets.make_blobs(n_samples=150, centers=4, n_features=2, cluster_std=0.1, center_box=(0,500))
clusest = ClusterEstimation(simulated_detections, False)
clusest.run()