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
from ..object_in_world import ObjectInWorld

# Placeholder pending finalisation of geolocation output format:
from ..cluster_estimation.detection_in_world import DetectionInWorld
 
class ClusterEstimation:
    """
    Estimate landing pad locations based on bounding box coordinates
    """
    def __init__(self, co_type="spherical", components=10, rand_state=0, weight_conc_prior=100, mean_precision_prior=1E-6,
                 min_points=100, new_points_per_run=10, weight_drop_threshold = 0.9):
        # Initalizes VMGMM model 
        self.__vgmm_model = sklearn.mixture.BayesianGaussianMixture(
            covariance_type = co_type, 
            n_components = components, 
            random_state = rand_state, 
            weight_concentration_prior = weight_conc_prior, 
            init_params = 'k-means++', 
            mean_precision_prior = mean_precision_prior)
        # Filter parameters
        self.__weight_drop_threshold = weight_drop_threshold
        # Points storage 
        self.__all_points = []
        self.__current_bucket = []
        # Requirements to decide to run
        self.__min_points = min_points
        self.__points_per_run = new_points_per_run
        self.__has_ran_once = False
    
    # EXPERIMENTAL: USING PREDICTIONS TO CHECK VALID CLUSTERS
    def filter_by_points_ownership(self, weights, covariances, clusters):
        results = self.__vgmm_model.predict(self.__all_points)

        # Filtering by each cluster's point ownership
        cluster_with_points, points_per_cluster = np.unique(results, return_counts=True)
        points_filtered_cluster = np.array([])
        points_filtered_weights = np.array([])
        points_filtered_covariances = np.array([])

        for idx in cluster_with_points:
            points_filtered_cluster = np.stack((points_filtered_cluster, clusters[idx]))
            points_filtered_weights = np.stack((points_filtered_cluster, clusters[idx]))
            points_filtered_covariances = np.stack((points_filtered_cluster, clusters[idx]))
        
        print("Ownership clustering")
        print(weights)
        print(covariances)
        print(clusters)
        print("-----------------------------------------------")


    def run(self, detections: "list[DetectionInWorld]", run_override:  bool) -> "tuple[bool, list[ObjectInWorld | None]]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run. 
        """

        # Decide to run
        if not run_override and not self.decide_to_run(detections):
            return False, None
        
        # Fit points and get cluster data 
        self.__vgmm_model = self.__vgmm_model.fit(self.__all_points)
        clusters = self.__vgmm_model.means_
        covariances = self.__vgmm_model.covariances_
        weights = self.__vgmm_model.weights_

        # print("Initial:")
        # print(weights)
        # print(covariances)
        # print(clusters)
        # print("-----------------------------------------------")

        # Sort weights from largest to smallest, along with corresponding covariances and means
        indices = np.argsort(weights)[::-1]
        weights = np.sort(weights)[::-1]
        temp_arr = np.array(covariances)
        for i in range(len(indices)):
            covariances[i] = temp_arr[indices[i]]
        temp_arr = np.array(clusters)
        for i in range(len(indices)):
            clusters[i] = temp_arr[indices[i]]

        # print("Sorted:")
        # print(weights)
        # print(covariances)
        # print(clusters)
        # print("-----------------------------------------------")

        # Loop through each cluster
        # clusters is a list of centers ordered by weights
        # most likely cluster listed first in descending weights order
        objects_in_world = []
        total_clusters = clusters.shape[0]
        num_viable_clusters = 1
        # Drop all clusters after a weight_drop_threshold drop in weight occured
        while num_viable_clusters < total_clusters:
            if (weights[num_viable_clusters-1] / weights[num_viable_clusters]) > (1 / self.__weight_drop_threshold):
                weights = weights[:num_viable_clusters]
                clusters = clusters[:num_viable_clusters]
                covariances = covariances[:num_viable_clusters]
                break
            num_viable_clusters += 1

        # print("Weight-filtered:")
        # print(weights)
        # print(covariances)
        # print(clusters)
        # print("-----------------------------------------------")

        # Bad detection removal
        i = 0
        while i < len(clusters):
            nearby_point = False
            j = 0
            while j < len(self.__all_points):
                # Remove cluster if nothing within 2 metres
                if self.get_distance(clusters[i], self.__all_points[j]) < 10:
                    nearby_point = True
                    break
                j += 1
            if not nearby_point:
                clusters = np.delete(clusters, i, 0)
                covariances = np.delete(covariances, i, 0)
            else:
                i += 1


        print("Bad clusters removed:")
        print(weights)
        print(covariances)
        print(clusters)
        print("-----------------------------------------------")
        print()
        print()

        # Create output list of remaining (valid) clusters 
        for i in range(len(clusters)):
            object_created, object_to_add = ObjectInWorld.create(
                clusters[i][0], 
                clusters[i][1], 
                covariances[i]
                )
            if object_created:
                objects_in_world.append(object_to_add)



        return True, objects_in_world

    def decide_to_run(self, detections: "list[DetectionInWorld]") -> bool:
        """
        Decide when to run cluster estimation model 
        """
        self.__current_bucket += self.convert_detections_to_point(detections)
        # Don't run if total points under minimum requirement
        if len(self.__all_points) + len(self.__current_bucket) < self.__min_points:
            return False
        # Don't run if not enough new points
        if len(self.__current_bucket) < self.__points_per_run and self.__has_ran_once == True:
            return False
        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []
        self.__has_ran_once = True

        return True
        
    def get_distance(self, cluster, point) -> float:
        """
        Get distance between a point and a cluster center
        """
        return np.sqrt(np.square(cluster[0]-point[0])+np.square(cluster[1]-point[1]))

    def convert_detections_to_point(self, detections: "list[DetectionInWorld]") -> list[float, float]:
        """
        Convert DetectionInWorld input object to a [x,y] position to store 
        """
        points = []
        for detection in detections:
            points.append([detection.location_x, detection.location_y])

        return points

