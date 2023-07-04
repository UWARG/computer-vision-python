"""
TODO: Write description
"""

import sklearn
import numpy as np

# Parent directory import not working
from modules.object_in_world import ObjectInWorld

# Placeholder:
from modules.cluster_estimation.detection_in_world import DetectionInWorld
 
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
        self.__current_bucket = []
        # Requirements to decide to run
        self.__min_points = min_points
        self.__points_per_run = new_points_per_run
        self.__no_points_count = 0
       
    def run(self, detections: "list[DetectionInWorld]", run_override:  bool) -> "tuple[bool, list[ObjectInWorld | None]]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run. 
        """

        if not run_override and not self.decide_to_run(detections):
            return False, None
        
        self.__vgmm_model = self.__vgmm_model.fit(self.__all_points)
        clusters = self.__vgmm_model.means_
        covariances = self.__vgmm_model.covariances_
        weights = self.__vgmm_model.weights_


        i = 0
        while i < len(clusters):
            nearby_point = False
            j = 0
            while j < len(self.__all_points):
                # Remove cluster if nothing within 2 metres
                if self.get_distance(clusters[i], self.__all_points[j]) < 2:
                    nearby_point = True
                    break
                j += 1
            if not nearby_point:
                weights = np.delete(weights, i, 0)
                clusters = np.delete(clusters, i, 0)
            else:
                i += 1



        indices = np.argsort(weights)[::-1]
        weights = np.sort(weights)[::-1]


        # Loop through each cluster
        # clusters is a list of centers ordered by weights
        # most likely cluster listed first in descending weights order
        total_clusters = clusters.shape[0]
        num_viable_clusters = 1

        objects_in_world = []
        # Drop all clusters after a 50% drop in weight occurs
        while num_viable_clusters < total_clusters:
            if (weights[num_viable_clusters] / weights[num_viable_clusters-1]) > 2:
                break
            # Append new ObjectInWorld to output list
            # TODO: Verify if current indexing is correct 
            object_created, object_to_add = ObjectInWorld.create(
                clusters[indices[num_viable_clusters]][0].items(), 
                clusters[indices[num_viable_clusters]][1].items(), 
                covariances[indices[num_viable_clusters]]
                )
            if object_created:
                objects_in_world.append(object_to_add)
            num_viable_clusters += 1
        # TODO: Decide when to not return any cluster centers
        return True, objects_in_world

    def decide_to_run(self, detections: "list[DetectionInWorld]") -> bool:
        """
        Decide when to run cluster estimation model 
        """
        self.__current_bucket += detections
        # Don't run if total points under minimum requirement
        if len(self.__all_points) + len(self.__current_bucket) < self.__min_points:
            return False
        # Don't run if not enough new points
        if self.__current_bucket < self.__points_per_run:
            return False
        # Both requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = 0
        return True
        
    def get_distance(cluster, point) -> float:
        return np.sqrt(np.square(cluster[0]-point[0])+np.square(cluster[1]-point[1]))


# Debug
if __name__ == "__main__":
    simulated_detections, y = sklearn.datasets.make_blobs(n_samples=150, centers=4, n_features=2, cluster_std=0.1, center_box=(0,500))
    clusest = ClusterEstimation(simulated_detections, False)
    clusest.run()