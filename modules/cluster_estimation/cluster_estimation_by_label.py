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
from ..cluster_estimation import cluster_estimation
from ..common.modules.logger import logger


class ClusterEstimationByLabel:
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

    METHODS
    -------
    run()
        Take in list of object detections and return list of estimated object locations
        if number of detections is sufficient, or if manually forced to run.

    cluster_by_label()
        Take in list of detections of the same label and return list of estimated object locations
        of the same label.

    __decide_to_run()
        Decide when to run cluster estimation model.

    __sort_by_weights()
        Sort input model output list by weights in descending order.

    __sort_by_labels()
        Sort input detection list by labels in descending order.

    __convert_detections_to_point()
        Convert DetectionInWorld input object to a [x,y] position to store.

    """

    __create_key = object()

    @classmethod
    def create(
        cls,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        cluster_model: cluster_estimation.ClusterEstimation,
        local_logger: logger.Logger,
    ) -> "tuple[bool, ClusterEstimationByLabel | None]":
        """
        Data requirement conditions for estimation model to run.
        """

        # At least 1 point for model to fit
        if min_activation_threshold < 1:
            return False, None

        return True, ClusterEstimationByLabel(
            cls.__create_key,
            min_activation_threshold,
            min_new_points_to_run,
            cluster_model,
            local_logger,
        )

    def __init__(
        self,
        class_private_create_key: object,
        min_activation_threshold: int,
        min_new_points_to_run: int,
        cluster_model: cluster_estimation.ClusterEstimation,
        local_logger: logger.Logger,
    ) -> None:
        """
        Private constructor, use create() method.
        """
        assert (
            class_private_create_key is ClusterEstimationByLabel.__create_key
        ), "Use create() method"

        # Points storage
        self.__all_points: "list[tuple[float, float]]" = []
        self.__current_bucket: "list[tuple[float, float]]" = []

        # Requirements to decide to run
        self.__min_activation_threshold = min_activation_threshold
        self.__min_new_points_to_run = min_new_points_to_run
        self.__logger = local_logger

        # cluster_model
        self.__cluster_model = cluster_model

    def run(
        self,
        detections: "list[detection_in_world.DetectionInWorld]",
        run_override: bool,
    ) -> "tuple[bool, list[object_in_world.ObjectInWorld] | None]":
        """
        Take in list of detections and return list of estimated object locations
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
        self.__current_bucket += detections
        self.__all_points = []

        # Decide to run
        if not self.__decide_to_run(run_override):
            return False, None

        # sort bucket by label in descending order
        self.__all_points = self.__sort_by_labels(self.__all_points)
        detections_in_world = []

        # init search parameters
        ptr = 0

        # itterates through all points
        while ptr < len(self.__all_points):
            # reference label
            label = (self.__all_points[ptr]).label

            # creates bucket of points with the same label since bucket is sorted by label
            bucket_labelled = []
            while ptr < len(self.__all_points) and (self.__all_points[ptr]).label == label:
                bucket_labelled.append(self.__all_points[ptr])
                ptr += 1

            # skip if no other objects have the same label
            if len(bucket_labelled) == 1:
                continue

            print("len bucket = "+str(len(bucket_labelled)))

            result, labelled_detections_in_world = self.__cluster_model.run(bucket_labelled, run_override)

            print("labelled detections = "+str(len(labelled_detections_in_world)))

            for object in labelled_detections_in_world:
                object.label = label
            
            # checks if cluster_by_label ran succssfully
            if not result:
                self.__logger.warning(
                    f"did not add objects of label={label} to total object detections"
                )
                continue

            detections_in_world += labelled_detections_in_world

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
            if count_current < self.__min_new_points_to_run:
                return False

        # No data can not run
        if count_all + count_current == 0:
            return False

        # Requirements met, empty bucket and run
        self.__all_points += self.__current_bucket
        self.__current_bucket = []

        return True

    @staticmethod
    def __sort_by_labels(
        points: "list[detection_in_world.DetectionInWorld]",
    ) -> "list[detection_in_world.DetectionInWorld]":
        """
        Sort input detection list by labels in descending order.

        PARAMETERS
        ----------
        detections: list[detection_in_world.DetectionInWorld]
            List containing detections.

        RETURNS
        -------
        list[tuple[np.ndarray, float, float]]
            List containing detection points sorted in descending order by label
        """
        return sorted(
            points, key=lambda x: x.label, reverse=True
        )  # the label is stored at index 2 of object

    @staticmethod
    def __convert_detections_to_point(
        detections: "list[detection_in_world.DetectionInWorld]",
    ) -> "list[tuple[float, float, int]]":
        """
        Convert DetectionInWorld input object to a list of points- (x,y) positions with label, to store.

        PARAMETERS
        ----------
        detections: list[DetectionInWorld]
            List of DetectionInWorld intermediate objects, the data structure that is passed to the
            worker.

        RETURNS
        -------
        points: list[tuple[float, float, int]]
            List of points (x,y) and their label
        -------
        """
        points = []

        # Input detections list is empty
        if len(detections) == 0:
            return points

        # Convert DetectionInWorld objects
        for detection in detections:
            # `centre` attribute holds positioning data
            points.append(tuple([detection.centre[0], detection.centre[1], detection.label]))

        return points
