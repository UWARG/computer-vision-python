"""
Cluster estimation by label.
"""

from . import cluster_estimation
from .. import detection_in_world
from .. import object_in_world
from ..common.modules.logger import logger


class ClusterEstimationByLabel:
    """
    Cluster estimation filtered on label.

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
        Cluster estimation filtered by label.
    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        max_num_components: int,
        random_state: int,
        local_logger: logger.Logger,
    ) -> "tuple[True, ClusterEstimationByLabel] | tuple[False, None]":
        """
        See `ClusterEstimation` for parameter descriptions.
        """

        is_valid_arguments = cluster_estimation.ClusterEstimation.check_create_arguments(
            min_activation_threshold, min_new_points_to_run, max_num_components, random_state
        )

        if not is_valid_arguments:
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

        # Construction arguments for `ClusterEstimation`
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
        input_detections: list[detection_in_world.DetectionInWorld],
        run_override: bool,
    ) -> tuple[True, dict[int, list[object_in_world.ObjectInWorld]]] | tuple[False, None]:
        """
        See `ClusterEstimation` for parameter descriptions.

        RETURNS
        -------
        model_ran: bool
            True if ClusterEstimation object successfully ran its estimation model, False otherwise.

        labels_to_objects: dict[int, list[object_in_world.ObjectInWorld] or None.
            Dictionary where the key is a label and the value is a list of all cluster detections with that label.
            ObjectInWorld objects don't have a label property, but they are sorted into label categories in the dictionary.
        """
        label_to_detections: dict[int, list[detection_in_world.DetectionInWorld]] = {}

        # Sorting detections by label
        for detection in input_detections:
            if not detection.label in label_to_detections:
                label_to_detections[detection.label] = []
            
            label_to_detections[detection.label].append(detection)

        labels_to_objects: dict[int, list[object_in_world.ObjectInWorld]] = {}

        for label, detections in label_to_detections.items():
            # Create cluster estimation for label if it doesn't exist
            if not label in self.__label_to_cluster_estimation_model:
                result, cluster_model = cluster_estimation.ClusterEstimation.create(
                    self.__min_activation_threshold,
                    self.__min_new_points_to_run,
                    self.__max_num_components,
                    self.__random_state,
                    self.__local_logger,
                )
                if not result:
                    self.__local_logger.error(
                        f"Failed to create cluster estimation for label {label}"
                    )
                    return False, None
                
                self.__label_to_cluster_estimation_model[label] = cluster_model

            # Runs cluster estimation for specific label
            result, clusters = self.__label_to_cluster_estimation_model[label].run(
                detections,
                run_override,
            )

            if not result:
                self.__local_logger.error(
                    f"Failed to run cluster estimation model for label {label}"
                )
                return False, None

            if not label in labels_to_objects:
                labels_to_objects[label] = []
            labels_to_objects[label] += clusters

        return True, labels_to_objects
