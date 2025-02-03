"""
Take in bounding box coordinates from Geolocation and use to estimate landing pad locations.
Returns an array of classes, each containing the x coordinate, y coordinate, and spherical 
covariance of each landing pad estimation.
"""

from .. import detection_in_world
from .. import object_in_world
from ..common.modules.logger import logger
from . import cluster_estimation


class ClusterEstimationByLabel:
    """
    Estimate landing pad locations based on landing pad ground detection. Estimation
    works by predicting 'cluster centres' from groups of closely placed landing pad
    detections.

    ATTRIBUTES
    ----------
    min_activation_threshold: int
        Minimum total data points before model runs. Must be at least max_num_components.

    min_new_points_to_run: int
        Minimum number of new data points that must be collected before running model.

    max_num_components: int
        Max number of real landing pads. Must be at least 1.

    random_state: int
        Seed for randomizer, to get consistent results.

    local_logger: Logger
        For logging error and debug messages.

    METHODS
    -------
    run()
        Take in list of object detections and return dictionary of labels to
        to corresponging clusters of estimated object locations if number of
        detections is sufficient, or if manually forced to run.
    """

    # pylint: disable=too-many-instance-attributes

    __create_key = object()

    @classmethod
    def create(
        cls,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        max_num_components: int,
        random_state: int,
        local_logger: logger.Logger,
    ) -> "tuple[bool, ClusterEstimationByLabel | None]":
        """
        Data requirement conditions for estimation model to run.
        """

        # At least 1 point for model to fit
        if min_activation_threshold < max_num_components:
            return False, None

        if min_new_points_to_run < 0:
            return False, None

        if max_num_components < 1:
            return False, None

        if random_state < 0:
            return False, None

        return True, ClusterEstimationByLabel(
            cls.__create_key,
            min_activation_threshold,
            min_new_points_to_run,
            max_num_components,
            random_state,
            local_logger,
        )

    def __init__(
        self,
        class_private_create_key: object,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        max_num_components: int,
        random_state: int,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert (
            class_private_create_key is ClusterEstimationByLabel.__create_key
        ), "Use create() method"

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__min_new_points_to_run = min_new_points_to_run
        self.__max_num_components = max_num_components
        self.__random_state = random_state
        self.__local_logger = local_logger

        # Cluster model corresponding to each label
        # Each cluster estimation object stores the detections given to in its __all_points bucket across runs
        self.__label_to_cluster_estimation_model: dict[
            int, cluster_estimation.ClusterEstimation
        ] = {}

    def run(
        self,
        input_detections: "list[detection_in_world.DetectionInWorld]",
        run_override: bool,
    ) -> "tuple[True, dict[int, list[object_in_world.ObjectInWorld]]] | tuple[False, None]":
        """
        Take in list of detections and return list of estimated object locations
        if number of detections is sufficient, or if manually forced to run.

        PARAMETERS
        ----------
        input_detections: list[DetectionInWorld]
            List containing DetectionInWorld objects which holds real-world positioning data to run
            clustering on.

        run_override: bool
            Forces ClusterEstimation to predict if data is available, regardless of any other
            requirements.

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation object successfully ran its estimation model, False otherwise.

        labels_to_object_clusters: dict[int, list[object_in_world.ObjectInWorld] or None.
            Dictionary where the key is a label and the value is a list of all cluster detections with that label
        """
        label_to_detections: dict[int, list[detection_in_world.DetectionInWorld]] = {}
        # Sorting detections by label
        for detection in input_detections:
            if not detection.label in label_to_detections:
                label_to_detections[detection.label] = []
            label_to_detections[detection.label].append(detection)

        labels_to_object_clusters: dict[int, list[object_in_world.ObjectInWorld]] = {}
        for label, detections in label_to_detections.items():
            # create cluster estimation for label if it doesn't exist
            if not label in self.__label_to_cluster_estimation_model:
                result, cluster_model = cluster_estimation.ClusterEstimation.create(
                    self.__min_activation_threshold,
                    self.__min_new_points_to_run,
                    self.__max_num_components,
                    self.__random_state,
                    self.__local_logger,
                    label,
                )
                if not result:
                    self.__local_logger.error(
                        f"Failed to create cluster estimation for label {label}"
                    )
                    return False, None
                self.__label_to_cluster_estimation_model[label] = cluster_model
            # runs cluster estimation for specific label
            result, clusters = self.__label_to_cluster_estimation_model[label].run(
                detections,
                run_override,
            )
            if not result:
                self.__local_logger.error(
                    f"Failed to run cluster estimation model for label {label}"
                )
                return False, None
            if not label in labels_to_object_clusters:
                labels_to_object_clusters[label] = []
            labels_to_object_clusters[label] += clusters

        return True, labels_to_object_clusters
