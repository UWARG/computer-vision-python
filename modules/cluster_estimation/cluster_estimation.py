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

    ATTRIBUTES
    ----------
    components: int, default = 10
        input number of clusters for model
    
    rand_state: int, default = 0
        seed for randomizer, to get consistent results
    
    min_activation_threshold: int, default = 100
        minimum total data points before model can be ran 

    new_points_per_run: int, default = 10
        minimum number of new data points that must be collected
        before running model 

    METHODS
    -------
    __init__()

    clear_all_data()

    reset_model()

    run()

    __decide_to_run()

    __get_distance()

    __convert_detections_to_point()

    __filter_by_points_ownership()
    
    __sort_by_weights()
    """
    # Model Hyperparams 
    __MODEL_INIT_PARAM = "k-means++"
    __COVAR_TYPE = "spherical"
    __WEIGHT_CONC_PRIOR = 100
    __MEAN_PRECISION_PRIOR = 1E-6
    __WEIGHT_DROP_THRESHOLD = 0.9
    __MAX_CLUSTER_SIZE = 2

    def __init__(self,
                 components=10,
                 rand_state=0,
                 min_activation_threshold=100,
                 new_points_per_run=10,):
        
        # Initalizes VGMM model
        self.__vgmm_model = sklearn.mixture.BayesianGaussianMixture(
            covariance_type = self.__COVAR_TYPE,
            n_components = components,
            random_state = rand_state,
            weight_concentration_prior = self.__WEIGHT_CONC_PRIOR,
            init_params = self.__MODEL_INIT_PARAM,
            mean_precision_prior = self.__MEAN_PRECISION_PRIOR)

        # Points storage
        self.__all_points = []
        self.__current_bucket = []

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__points_per_run = new_points_per_run
        self.__has_ran_once = False
    
    def clear_all_data(self):
        """
        Deletes all current points stored and emptys the current bucket
        """
        self.__all_points = []
        self.__current_bucket = []
        self.__has_ran_once = False

    def reset_model(self):
        """
        Resets model back to original initialization, before any data has been fitted
        """
        self.__vgmm_model = sklearn.base.clone(self.__vgmm_model)

    # EXPERIMENTAL: USING PREDICTIONS TO CHECK VALID CLUSTERS
    def __filter_by_points_ownership(self, weights, covariances, clusters):
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
    
    def __sort_by_weights(self, cluster_weight_covariances:tuple[np.ndarray, np.float64, np.float64]) -> tuple[np.ndarray, np.float64, np.float64]:
        # indices = np.argsort(weights)[::-1]
        # weights = np.sort(weights)[::-1]
        # temp_arr = np.array(covariances)
        # for i in range(len(indices)):
        #     covariances[i] = temp_arr[indices[i]]
        # temp_arr = np.array(clusters)
        # for i in range(len(indices)):
        #     clusters[i] = temp_arr[indices[i]]
        raise NotImplementedError

    def run(self, detections: "list[DetectionInWorld]", run_override: bool) -> "tuple[bool, list[ObjectInWorld | None]]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            list containing DetectionInWorld objects containing real-world positioning data to run clustering on
        
        run_override: bool, default = False 
            forces ClusterEstimation worker to run and output predictions regardless of other conditions

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation worker ran its estimation model

        objects_in_world: list[ObjectInWorld] or None
            list containing ObjectInWorld objects, containing position and covariance value. None if conditions not met
            and model not ran
        """
        # Store new input data
        self.__current_bucket += self.__convert_detections_to_point(detections)

        # Decide to run
        if not run_override and not self._decide_to_run(detections):
            return False, None
        
        # Fit points and get cluster data
        self.__vgmm_model = self.__vgmm_model.fit(self.__all_points)
        converged = self.__vgmm_model.converged_
        clusters = self.__vgmm_model.means_
        covariances = self.__vgmm_model.covariances_
        weights = self.__vgmm_model.weights_

        model_output = list(zip(clusters, weights, covariances))

        print("Initial:")
        print(weights)
        print(covariances)
        print(clusters)
        print("-----------------------------------------------")

        # Sort weights from largest to smallest, along with corresponding covariances and means
        # Tuple comes sorted ONLY IF MODEL CONVERGES
        if not converged:
            self.__sort_by_weights()
        
        print("Sorted:")
        print(weights)
        print(covariances)
        print(clusters)
        print("-----------------------------------------------")

        # Loop through each cluster and drop all clusters after a __WEIGHT_DROP_THRESHOLD drop in weight occured
        num_viable_clusters = 1
        weight_drop_thresh_reached = False
        while num_viable_clusters < clusters.shape[0] and not weight_drop_thresh_reached:
            # Check if weight dropped occured between previous cluster and current cluster
            if (model_output[1][num_viable_clusters-1] / model_output[1][num_viable_clusters]) > (1 / self.__WEIGHT_DROP_THRESHOLD):
                model_output = model_output[:num_viable_clusters]
                weight_drop_thresh_reached = True

            num_viable_clusters += 1

        print("Weight-filtered:")
        print(weights)
        print(covariances)
        print(clusters)
        print("-----------------------------------------------")

        # Bad detection removal
        i = 0
        while i < len(clusters):
            nearby_point = False
            j = 0
            while j < len(self.__all_points):
                # Remove cluster if nothing within __MAX_CLUSTER_SIZE metres
                if self.__get_distance(clusters[i], self.__all_points[j]) < self.__MAX_CLUSTER_SIZE:
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

        # Create output list of remaining valid clusters 
        detections_in_world = []
        for i in range(len(clusters)):
            result_created, result_to_add = ObjectInWorld.create(clusters[i][0],
                                                                 clusters[i][1],
                                                                 covariances[i],)
            if result_created:
                detections_in_world.append(result_to_add)

        return True, detections_in_world

    def __decide_to_run(self, detections: "list[DetectionInWorld]") -> bool:
        """
        Decide when to run cluster estimation model 
        """
        # Don't run if total points under minimum requirement
        if len(self.__all_points) + len(self.__current_bucket) < self.__min_activation_threshold:
            return False
        
        # Don't run if not enough new points
        if len(self.__current_bucket) < self.__points_per_run and self.__has_ran_once == True:
            return False
        
        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []
        self.__has_ran_once = True

        return True
        
    def __get_distance(self, cluster, point) -> float:
        """
        Get distance between a point and a cluster center
        """
        return np.linalg.norm(cluster - point)

    def __convert_detections_to_point(self, detections: "list[DetectionInWorld]") -> list[tuple(float, float)]:
        """
        Convert DetectionInWorld input object to a [x,y] position to store 
        """
        points = []
        for detection in detections:
            points.append((detection.location_x, detection.location_y))

        return points