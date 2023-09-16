"""
Take in bounding box coordinates from Geolocation and use to estimate landing pad locations.
Returns an array of classes, each containing the x coordinate, y coordinate, and spherical 
covariance of each landing pad estimation.
"""

import sklearn
import sklearn.datasets
import sklearn.mixture
import numpy as np

from ..object_in_world import ObjectInWorld
from ..detection_in_world import DetectionInWorld


class ClusterEstimation:
    """
    Estimate landing pad locations based on landing pad ground detection. Estimation
    works by predicting 'cluster centers' from groups of closely placed landing pad
    detections.

    ATTRIBUTES
    ----------
    components: int, default = 10
        Input number of clusters for model.
    
    rand_state: int, default = 0
        Seed for randomizer, to get consistent results.
    
    min_activation_threshold: int, default = 100
        Minimum total data points before model can be ran.

    new_points_per_run: int, default = 10
        Minimum number of new data points that must be collected
        before running model.

    METHODS
    -------
    __init__()
        Sets cluster model run conditions and random state.

    clear_all_data()
        Clears all past accumulated data.

    reset_model()
        Unfits the model of any data.

    run()
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

    __decide_to_run()
        Decide when to run cluster estimation model.

    __convert_detections_to_point()
        Convert DetectionInWorld input object to a [x,y] position to store.
    
    __sort_by_weights()
        Sort input model output list by weights in descending order.

    __get_distance()
        Calculates distance between a point and a cluster center.
    """
    __create_key = object()
    
    # VGMM Hyperparameters
    __MODEL_INIT_PARAM = "k-means++"
    __COVAR_TYPE = "spherical"
    __WEIGHT_CONCENTRACTION_PRIOR = 100
    __MEAN_PRECISION_PRIOR = 1E-6
    __MAX_MODEL_ITERATIONS = 1000

    # Hyperparameters to clean up model outputs
    __WEIGHT_DROP_THRESHOLD = 0.1
    __MAX_COVARIANCE_THRESHOLD = 10

    # Real-world scenario Hyperparameters
    __MAX_NUM_COMPONENTS = 10  # assumed maximum number of real landing pads

    @classmethod
    def create(cls,
               min_activation_threshold:int,
               min_new_points_to_run:int,
               random_state:int):
        """
        Data requirement conditions for estimation worker to run.
        """
        if min_activation_threshold < 0 or min_new_points_to_run < 0 or random_state < 0:
            return False, None
        
        return True, ClusterEstimation(cls.__create_key, min_activation_threshold, min_new_points_to_run, random_state)


    def __init__(self,
                 class_private_create_key,
                 min_activation_threshold,
                 min_new_points_to_run,
                 random_state):
        """
        Private constructor, use create() method.
        """
        assert (class_private_create_key is ClusterEstimation.__create_key), "Use create() method"

        # Initalizes VGMM
        self.__vgmm = sklearn.mixture.BayesianGaussianMixture(
            covariance_type = self.__COVAR_TYPE,
            n_components = self.__MAX_NUM_COMPONENTS,
            random_state = random_state,
            weight_concentration_prior = self.__WEIGHT_CONCENTRACTION_PRIOR,
            init_params = self.__MODEL_INIT_PARAM,
            mean_precision_prior = self.__MEAN_PRECISION_PRIOR,
            max_iter=self.__MAX_MODEL_ITERATIONS)

        # Points storage
        self.__all_points = []
        self.__current_bucket = []

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__min_new_points_to_run = min_new_points_to_run
        self.__has_ran_once = False
    
    def clear_all_data(self):
        """
        Deletes all current points stored and emptys the current bucket.
        """
        self.__all_points = []
        self.__current_bucket = []
        self.__has_ran_once = False

    def reset_model(self):
        """
        Resets model back to original initialization, before any data has been fitted.
        """
        self.__vgmm = sklearn.base.clone(self.__vgmm)

    def run(self, detections: "list[DetectionInWorld]", run_override: bool) -> "tuple[bool, list[ObjectInWorld | None]]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            List containing DetectionInWorld objects containing real-world positioning data to run clustering on.
        
        run_override: bool, default = False 
            Forces ClusterEstimation worker to run and output predictions regardless of other conditions.

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation worker ran its estimation model.

        objects_in_world: list[ObjectInWorld] or None.
            List containing ObjectInWorld objects, containing position and covariance value. None if conditions not met
            and model not ran.
        """
        # Store new input data
        self.__current_bucket += self.__convert_detections_to_point(detections)

        # Decide to run
        if not run_override and not self.__decide_to_run():
            return False, None
        
        # Fit points and get cluster data
        self.reset_model() 
        self.__vgmm = self.__vgmm.fit(self.__all_points)

        model_output:list[np.array, float, float] = list(zip(self.__vgmm.means_,
                                                            self.__vgmm.weights_,
                                                            self.__vgmm.covariances_))
        # Empty cluster removal
        model_output = self.__filter_by_points_ownership(model_output)

        # Sort weights from largest to smallest, along with corresponding covariances and means
        # TODO: Tuple should come already sorted if model converges, but appears to not come sort in one
        # of the tests, hence sorting is done every call. Investigate behaviour exactly to only sort when
        # necessary
        model_output = self.__sort_by_weights(model_output)

        # Loop through each cluster and drop all clusters after a __WEIGHT_DROP_THRESHOLD drop in weight occured
        viable_cluster_counts = 1
        weight_drop_thresh_reached = False
        while viable_cluster_counts < len(model_output) and not weight_drop_thresh_reached:
            # Check if weight dropped occured between previous cluster and current cluster
            # index [1] after to access weights values
            if (model_output[viable_cluster_counts][1] / model_output[viable_cluster_counts-1][1]
                < self.__WEIGHT_DROP_THRESHOLD):
                model_output = model_output[:viable_cluster_counts]
                weight_drop_thresh_reached = True

            viable_cluster_counts += 1
        
        # Remove clusters with covariances too large
        model_output = self.__filter_by_covariances(model_output)

        # Iterate through model_output for each cluster center
        # i = 0
        # while i < len(model_output):
        #     valid_point = False
        #     j = 0
        #     while j < len(self.__all_points):
        #         # Cluster only valid if at least 1 point falls within 5m
        #         if (self.__get_distance(model_output[i][0], self.__all_points[j]) < self.__MAX_CLUSTER_SIZE):
        #             valid_point = True
        #         j += 1

        #     # remove point if not valid
        #     if not valid_point:
        #         model_output.pop(i)
        #     else:
        #         i += 1

        # Create output list of remaining valid clusters
        detections_in_world = []
        for i in range(len(model_output)):
            result_created, result_to_add = ObjectInWorld.create(model_output[i][0][0],
                                                                 model_output[i][0][1],
                                                                 model_output[i][2],)
            if result_created:
                detections_in_world.append(result_to_add)

        return True, detections_in_world

    def __decide_to_run(self) -> bool:
        """
        Decide when to run cluster estimation model.
        """
        # Don't run if total points under minimum requirement
        if len(self.__all_points) + len(self.__current_bucket) < self.__min_activation_threshold:
            return False
        
        # Don't run if not enough new points
        if len(self.__current_bucket) < self.__min_new_points_to_run and self.__has_ran_once == True:
            return False
        
        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []
        self.__has_ran_once = True

        return True
    
    def __sort_by_weights(self, model_output:list[np.ndarray,
                                                                np.float64,
                                                                np.float64]) -> list[np.ndarray, np.float64, np.float64]:
        """
        Sort input model output list by weights in descending order.
        """
        return sorted(model_output, key = lambda x:x[1], reverse=True)
    
    def __get_distance(self, point, cluster):
        """
        Calculates distance between a point and a cluster center.
        """
        diff_vector = [0,0]
        diff_vector[0] = point[0] - cluster[0]
        diff_vector[1] = point[1] - cluster[1]
        
        return np.linalg.norm(diff_vector)

    def __convert_detections_to_point(self, detections: "list[DetectionInWorld]") -> list[tuple([float, float])]:
        """
        Convert DetectionInWorld input object to a [x,y] position to store.
        """
        points = []
        for detection in detections:
            points.append(tuple([detection.centre[0], detection.centre[1]]))

        return points
    
    def __filter_by_points_ownership(self, 
                                     model_output:list[np.ndarray,
                                                       np.float64,
                                                       np.float64]) -> list[np.ndarray,
                                                                            np.float64,
                                                                            np.float64]:
        """
        Removes any clusters that don't have any points belonging to it.

        PARAMETERS
        ----------

        RETURNS
        -------
        """
        
        results = self.__vgmm.predict(self.__all_points)
        valid_clusters = []

        
        # Filtering by each cluster's point ownership
        unique_clusters, num_points_per_cluster = np.unique(results, return_counts=True)
        # Remove empty clusters
        i = 0
        for i in range(len(model_output)):
            if (i in unique_clusters):
                valid_clusters.append(model_output[i])


        return valid_clusters

    def __filter_by_covariances(self,
                                model_output:list[np.ndarray,
                                                  np.float64,
                                                  np.float64]) -> list[np.ndarray,
                                                                       np.float64,
                                                                       np.float64]:
        """
        Removes any cluster with covariances much higher than lowest covariance value
        """
        # python list and not np array, need to loop through manually
        min_covariance = 1E6
        for item in model_output:
            if (min_covariance > item[2]):
                min_covariance = item[2]
        max_covariance_threshold = min_covariance * self.__MAX_COVARIANCE_THRESHOLD

        # Iterate through model_output for each cluster center
        i = 0
        while i < len(model_output):
            valid_cluster = False
            if model_output[i][2] > max_covariance_threshold:
                model_output.pop(i)
            else:
                i += 1

        return model_output