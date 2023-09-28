"""
Take in bounding box coordinates from Geolocation and use to estimate landing pad locations.
Returns an array of classes, each containing the x coordinate, y coordinate, and spherical 
covariance of each landing pad estimation.
"""

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.mixture

from .. import object_in_world
from .. import detection_in_world


class ClusterEstimation:
    """
    Estimate landing pad locations based on landing pad ground detection. Estimation
    works by predicting 'cluster centers' from groups of closely placed landing pad
    detections.

    ATTRIBUTES
    ----------
    random_state: int
        Seed for randomizer, to get consistent results.
    
    min_activation_threshold: int
        Minimum total data points before model runs.

    new_points_per_run: int
        Minimum number of new data points that must be collected
        before running model.

    METHODS
    -------
    reset_model()
        Unfits the model of any data.

    run()
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

    __decide_to_run()
        Decide when to run cluster estimation model.

    __sort_by_weights()
        Sort input model output list by weights in descending order.

    __get_distance()
        Calculates distance between a point and a cluster center.

    __convert_detections_to_point()
        Convert DetectionInWorld input object to a [x,y] position to store.

    __filter_by_points_ownership()
        Removes any clusters that don't have any points belonging to it.

    __filter_by_covariances()
        Removes any cluster with covariances much higher than the lowest covariance value.
    """
    __create_key = object()

    # VGMM Hyperparameters
    __MODEL_INIT_PARAM = "k-means++"
    __COVAR_TYPE = "spherical"
    __WEIGHT_CONCENTRATION_PRIOR = 100
    __MEAN_PRECISION_PRIOR = 1E-6
    __MAX_MODEL_ITERATIONS = 1000

    # Hyperparameters to clean up model outputs
    __WEIGHT_DROP_THRESHOLD = 0.1
    __MAX_COVARIANCE_THRESHOLD = 10

    # Real-world scenario Hyperparameters
    __MAX_NUM_COMPONENTS = 10  # assumed maximum number of real landing pads

    @classmethod
    def create(cls,
               min_activation_threshold: int,
               min_new_points_to_run: int,
               random_state: int) -> "tuple[bool, ClusterEstimation | None]":
        """
        Data requirement conditions for estimation worker to run.
        """
        if min_activation_threshold < 0 or min_new_points_to_run < 0 or random_state < 0:
            return False, None

        return True, ClusterEstimation(cls.__create_key,
            min_activation_threshold,
            min_new_points_to_run,
            random_state,
        )

    def __init__(self,
                 class_private_create_key: object,
                 min_activation_threshold: int,
                 min_new_points_to_run: int,
                 random_state: int):
        """
        Private constructor, use create() method.
        """
        assert (class_private_create_key is ClusterEstimation.__create_key), "Use create() method"

        # Initializes VGMM
        self.__vgmm = sklearn.mixture.BayesianGaussianMixture(
            covariance_type=self.__COVAR_TYPE,
            n_components=self.__MAX_NUM_COMPONENTS,
            random_state=random_state,
            weight_concentration_prior=self.__WEIGHT_CONCENTRATION_PRIOR,
            init_params=self.__MODEL_INIT_PARAM,
            mean_precision_prior=self.__MEAN_PRECISION_PRIOR,
            max_iter=self.__MAX_MODEL_ITERATIONS,
        )

        # Points storage
        self.__all_points = []
        self.__current_bucket = []

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__min_new_points_to_run = min_new_points_to_run
        self.__has_ran_once = False

    def run(self, detections: "list[detection_in_world.DetectionInWorld]", run_override: bool) \
        -> "tuple[bool, list[ObjectInWorld] | None]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            List containing DetectionInWorld objects containing real-world positioning data to run
            clustering on.
        
        run_override: bool
            Forces ClusterEstimation worker to run and output predictions regardless of other
            conditions.

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation worker ran its estimation model.

        objects_in_world: list[ObjectInWorld] or None.
            List containing ObjectInWorld objects, containing position and covariance value.
            None if conditions not met and model not ran.
        """
        # Store new input data
        self.__current_bucket += self.__convert_detections_to_point(detections)

        # Decide to run
        if not run_override and not self.__decide_to_run():
            return False, None

        # Fit points and get cluster data
        self.__vgmm = self.__vgmm.fit(self.__all_points)
        model_output: "list[np.array, float, float]" = list(
            zip(
                self.__vgmm.means_,
                self.__vgmm.weights_,
                self.__vgmm.covariances_,
            )
        )

        # Empty cluster removal
        model_output = self.__filter_by_points_ownership(model_output)

        # Sort weights from largest to smallest, along with corresponding covariances and means
        model_output = self.__sort_by_weights(model_output)

        # Filter out all clusters after __WEIGHT_DROP_THRESHOLD weight drop occurs
        viable_clusters = [model_output[0]]
        for i in range(1, len(model_output)):
            if model_output[i][1] / model_output[i - 1][1] < self.__WEIGHT_DROP_THRESHOLD:
                break

            viable_clusters.append(model_output[i])

        # Remove clusters with covariances too large
        model_output = self.__filter_by_covariances(model_output)

        # Create output list of remaining valid clusters
        detections_in_world = []
        for cluster in model_output:
            result, landing_pad = object_in_world.ObjectInWorld.create(
                cluster[0][0],
                cluster[0][1],
                cluster[2],
            )

            if result:
                detections_in_world.append(landing_pad)

        return True, detections_in_world

    def __decide_to_run(self) -> bool:
        """
        Decide when to run cluster estimation model.
        """
        # Don't run if total points under minimum requirement
        if len(self.__all_points) + len(self.__current_bucket) < self.__min_activation_threshold:
            return False

        # Don't run if not enough new points
        if len(self.__current_bucket) < self.__min_new_points_to_run and self.__has_ran_once:
            return False

        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []
        self.__has_ran_once = True

        return True

    @staticmethod
    def __sort_by_weights(model_output: "list[np.ndarray, float, float]") \
        -> "list[tuple[np.ndarray, float, float]]":
        """
        Sort input model output list by weights in descending order.

        PARAMETERS
        ----------
        model_output: list[np.ndarray, float, float]
            List containing predicted cluster centers, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        list[np.ndarray, float, float]
            List containing predicted cluster centers sorted by weights in descending order.
        """
        return sorted(model_output, key=lambda x: x[1], reverse=True)

    @staticmethod
    def __get_distance(point, cluster):
        """
        Calculates distance between a point and a cluster center.

        PARAMETERS
        ----------
        point: np.array
            Coordinate position of point [x position, y position].

        cluster: np.array
            Coordinate position of cluster [x position, y position].

        RETURNS
        -------
        distance: float
            Distance between the point and cluster.
        """
        diff_vector = [point[0] - cluster[0], point[1] - cluster[1]]

        return np.linalg.norm(diff_vector)

    @staticmethod
    def __convert_detections_to_point(detections: "list[detection_in_world.DetectionInWorld]") \
        -> "list[tuple[float, float]]":
        """
        Convert DetectionInWorld input object to a list of points- (x,y) positions, to store.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            List of DetectionInWorld intermediate objects, the data structure that is passed to the
            worker.

        RETURNS
        -------
        points: list[tuple[float, float]]
            List of points (x,y).
        -------
        """
        points = []
        for detection in detections:
            # attribute centre holds positioning data
            points.append(tuple([detection.centre[0], detection.centre[1]]))

        return points

    def __filter_by_points_ownership(self,
                                     model_output: "list[tuple[np.ndarray, float, float]]") \
        -> "list[tuple[np.ndarray, float, float]]":
        """
        Removes any clusters that don't have any points belonging to it.

        PARAMETERS
        ----------
        model_output: list[np.ndarray, float, float]
            List containing predicted cluster centers, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        filtered_output: list[np.ndarray, float, float]
            List containing predicted cluster centers after filtering.
        """

        results = self.__vgmm.predict(self.__all_points)
        filtered_output = []

        # Filtering by each cluster's point ownership
        unique_clusters, num_points_per_cluster = np.unique(results, return_counts=True)
        # Remove empty clusters
        for i in range(len(model_output)):
            if i in unique_clusters:
                filtered_output.append(model_output[i])

        return filtered_output

    def __filter_by_covariances(self, model_output: "list[tuple[np.ndarray, float, float]]") \
        -> "list[tuple[np.ndarray, float, float]]":
        """
        Removes any cluster with covariances much higher than the lowest covariance value.

        PARAMETERS
        ----------
        model_output: list[np.ndarray, float, float]
            List containing predicted cluster centers, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        list[np.ndarray, float, float]
            List containing predicted cluster centers after filtering by covariance.
        """
        # python list and not np array, need to loop through manually
        min_covariance = float("inf")
        for item in model_output:
            if item[2] < min_covariance:
                min_covariance = item[2]

        max_covariance_threshold = min_covariance * self.__MAX_COVARIANCE_THRESHOLD

        # Filter
        filtered_output = []
        for cluster in model_output:
            if cluster[2] <= max_covariance_threshold:
                filtered_output.append(cluster)

        return filtered_output
