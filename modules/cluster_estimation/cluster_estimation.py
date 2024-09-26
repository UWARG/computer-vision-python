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
    works by predicting 'cluster centres' from groups of closely placed landing pad
    detections.

    ATTRIBUTES
    ----------
    min_activation_threshold: int
        Minimum total data points before model runs.

    min_new_points_to_run: int
        Minimum number of new data points that must be collected before running model.

    random_state: int
        Seed for randomizer, to get consistent results.

    METHODS
    -------
    run()
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

    __decide_to_run()
        Decide when to run cluster estimation model.

    __sort_by_weights()
        Sort input model output list by weights in descending order.

    __convert_detections_to_point()
        Convert DetectionInWorld input object to a [x,y] position to store.

    __filter_by_points_ownership()
        Removes any clusters that don't have any points belonging to it.

    __filter_by_covariances()
        Removes any cluster with covariances much higher than the lowest covariance value.
    """

    __create_key = object()

    # VGMM Hyperparameters
    __COVAR_TYPE = "spherical"
    __MODEL_INIT_PARAM = "k-means++"
    __WEIGHT_CONCENTRATION_PRIOR = 100
    __MEAN_PRECISION_PRIOR = 1e-6
    __MAX_MODEL_ITERATIONS = 1000

    # Real-world scenario Hyperparameters
    __MAX_NUM_COMPONENTS = 10  # assumed maximum number of real landing pads

    # Hyperparameters to clean up model outputs
    __WEIGHT_DROP_THRESHOLD = 0.1
    __MAX_COVARIANCE_THRESHOLD = 10

    @classmethod
    def create(
        cls, min_activation_threshold: int, min_new_points_to_run: int, random_state: int
    ) -> "tuple[bool, ClusterEstimation | None]":
        """
        Data requirement conditions for estimation model to run.
        """
        # These parameters must be positive
        if min_new_points_to_run < 0 or random_state < 0:
            return False, None

        # At least 1 point for model to fit
        if min_activation_threshold < 1:
            return False, None

        return True, ClusterEstimation(
            cls.__create_key,
            min_activation_threshold,
            min_new_points_to_run,
            random_state,
        )

    def __init__(
        self,
        class_private_create_key: object,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        random_state: int,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_private_create_key is ClusterEstimation.__create_key, "Use create() method"

        # Initializes VGMM
        self.__vgmm = sklearn.mixture.BayesianGaussianMixture(
            covariance_type=self.__COVAR_TYPE,
            n_components=self.__MAX_NUM_COMPONENTS,
            init_params=self.__MODEL_INIT_PARAM,
            weight_concentration_prior=self.__WEIGHT_CONCENTRATION_PRIOR,
            mean_precision_prior=self.__MEAN_PRECISION_PRIOR,
            max_iter=self.__MAX_MODEL_ITERATIONS,
            random_state=random_state,
        )

        # Points storage
        self.__all_points: "list[tuple[float, float]]" = []
        self.__current_bucket: "list[tuple[float, float]]" = []

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__min_new_points_to_run = min_new_points_to_run
        self.__has_ran_once = False

    def run(
        self, detections: "list[detection_in_world.DetectionInWorld]", run_override: bool
    ) -> "tuple[bool, object_in_world.ObjectInWorld | None]":
        """
        Take in list of landing pad detections and return list of estimated landing pad locations
        if number of detections is sufficient, or if manually forced to run.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            List containing DetectionInWorld objects which holds real-world positioning data to run
            clustering on.

        run_override: bool
            Forces ClusterEstimation to predict if data is available, regardless of any other
            requirements.

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation object successfully ran its estimation model, False otherwise.

        objects_in_world: list[ObjectInWorld] or None.
            List containing ObjectInWorld objects, containing position and covariance value.
            None if conditions not met and model not ran or model failed to converge.
        """
        # Store new input data
        self.__current_bucket += self.__convert_detections_to_point(detections)

        # Decide to run
        if not self.__decide_to_run(run_override):
            return False, None

        # Fit points and get cluster data
        self.__vgmm = self.__vgmm.fit(self.__all_points)  # type: ignore

        # Check convergence
        if not self.__vgmm.converged_:
            return False, None

        # Get predictions from cluster model
        model_output: "list[tuple[np.ndarray, float, float]]" = list(
            zip(
                self.__vgmm.means_,  # type: ignore
                self.__vgmm.weights_,  # type: ignore
                self.__vgmm.covariances_,  # type: ignore
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

        model_output = viable_clusters

        # Remove clusters with covariances too large
        model_output = self.__filter_by_covariances(model_output)

        # Create output list of remaining valid clusters
        detections_in_world = []
        for cluster in model_output:

            result, object = object_in_world.ObjectInWorld.create(
                cluster[0][0],
                cluster[0][1],
                cluster[2],
                "object"
            )

            if result:
                detections_in_world.append([object])

        return True, detections_in_world

    def __decide_to_run(self, run_override: bool) -> bool:
        """
        Decide when to run cluster estimation model.

        PARAMETERS
        ----------
        run_override: bool
            Forces ClusterEstimation to predict if data is available, regardless of any other
            requirements.

        RETURNS
        -------
        bool
            True if estimation model will be run, False otherwise.
        """
        count_all = len(self.__all_points)
        count_current = len(self.__current_bucket)

        if not run_override:
            # Don't run if total points under minimum requirement
            if count_all + count_current < self.__min_activation_threshold:
                return False

            # Don't run if not enough new points
            if count_current < self.__min_new_points_to_run and self.__has_ran_once:
                return False

        # No data can not run
        if count_all + count_current == 0:
            return False

        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []
        self.__has_ran_once = True

        return True

    @staticmethod
    def __sort_by_weights(
        model_output: "list[tuple[np.ndarray, float, float]]",
    ) -> "list[tuple[np.ndarray, float, float]]":
        """
        Sort input model output list by weights in descending order.

        PARAMETERS
        ----------
        model_output: list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres sorted by weights in descending order.
        """
        return sorted(model_output, key=lambda x: x[1], reverse=True)

    @staticmethod
    def __convert_detections_to_point(
        detections: "list[detection_in_world.DetectionInWorld]",
    ) -> "list[tuple[float, float]]":
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

        # Input detections list is empty
        if len(detections) == 0:
            return points

        # Convert DetectionInWorld objects
        for detection in detections:
            # `centre` attribute holds positioning data
            points.append(tuple([detection.centre[0], detection.centre[1]]))

        return points

    def __filter_by_points_ownership(
        self, model_output: "list[tuple[np.ndarray, float, float]]"
    ) -> "list[tuple[np.ndarray, float, float]]":
        """
        Removes any clusters that don't have any points belonging to it.

        PARAMETERS
        ----------
        model_output: list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        filtered_output: list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres after filtering.
        """
        # List of each point's cluster index
        cluster_assignment = self.__vgmm.predict(self.__all_points)  # type: ignore

        # Find which cluster indices have points
        clusters_with_points = np.unique(cluster_assignment)

        # Remove empty clusters
        filtered_output: "list[tuple[np.ndarray, float, float]]" = []
        # By cluster index
        # pylint: disable-next=consider-using-enumerate
        for i in range(len(model_output)):
            if i in clusters_with_points:
                filtered_output.append(model_output[i])

        return filtered_output

    def __filter_by_covariances(
        self, model_output: "list[tuple[np.ndarray, float, float]]"
    ) -> "list[tuple[np.ndarray, float, float]]":
        """
        Removes any cluster with covariances much higher than the lowest covariance value.

        PARAMETERS
        ----------
        model_output: list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres, with each element having the format
            [(x position, y position), weight, covariance)].

        RETURNS
        -------
        list[tuple[np.ndarray, float, float]]
            List containing predicted cluster centres after filtering by covariance.
        """
        # Python list and not np array, need to loop through manually
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
