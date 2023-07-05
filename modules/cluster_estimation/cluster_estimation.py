"""
Take in bounding box coordinates from Geolocation and use to estimate landing pad locations.
Returns an array of classes, each containing the x coordinate, y coordinate, and spherical covariance
of each landing pad estimation. Whether or not a landing pad estimation is returned depends on the
drop of its mean compared to the next greatest mean.
"""

import sklearn
import sklearn.datasets
import sklearn.mixture
import numpy as np
# Parent directory import not working
from object_in_world import ObjectInWorld

# Placeholder:
from detection_in_world import DetectionInWorld
 
class ClusterEstimation:
    """
    Estimate landing pad locations based on bounding box coordinates
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


        print(weights)

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


        print(weights)
        indices = np.argsort(weights)[::-1]
        weights = np.sort(weights)[::-1]
        print(weights)
        print(indices)
        print(clusters)

        # Loop through each cluster
        # clusters is a list of centers ordered by weights
        # most likely cluster listed first in descending weights order
        

        objects_in_world = []
        # Return, at the very least, the cluster with the highest mean
        object_created, object_to_add = ObjectInWorld.create(
            clusters[indices[0]][0], 
            clusters[indices[0]][1], 
            covariances[indices[0]]
            )
        if object_created:
            objects_in_world.append(object_to_add)
        
        total_clusters = clusters.shape[0]
        num_viable_clusters = 1
        # Drop all clusters after a 50% drop in weight occurs
        while num_viable_clusters < total_clusters:
            if (weights[num_viable_clusters] / weights[num_viable_clusters-1]) > 2:
                break
            # Append new ObjectInWorld to output list
            # TODO: Verify if current indexing is correct 
            object_created, object_to_add = ObjectInWorld.create(
                clusters[indices[num_viable_clusters]][0], 
                clusters[indices[num_viable_clusters]][1], 
                covariances[indices[num_viable_clusters]]
                )
            if object_created:
                objects_in_world.append(object_to_add)
            num_viable_clusters += 1
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
        if len(self.__current_bucket) < self.__points_per_run:
            return False
        # Both requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = 0
        return True
        
    def get_distance(self, cluster, point) -> float:
        return np.sqrt(np.square(cluster[0]-point[0])+np.square(cluster[1]-point[1]))


# Debug
if __name__ == "__main__":
    simulated_detections, y = sklearn.datasets.make_blobs(n_samples=150, centers=4, n_features=2, cluster_std=0.1, center_box=(0,500))
    clusest = ClusterEstimation()
    ret_bool, ret_list = clusest.run(np.ndarray.tolist(simulated_detections), False)
    print(ret_bool)
    print(len(ret_list))
    for i in ret_list:
        print("x: %f, y: %f, var: %f" % (i.position_x, i.position_y, i.spherical_variance))