"""
Testing ClusterEstimationByLabel.
"""

import random
import numpy as np
import pytest
import sklearn.datasets

from modules.cluster_estimation import cluster_estimation_by_label
from modules.common.modules.logger import logger
from modules import detection_in_world

MIN_TOTAL_POINTS_THRESHOLD = 100
MIN_NEW_POINTS_TO_RUN = 10
MAX_NUM_COMPONENTS = 10
RNG_SEED = 0
CENTRE_BOX_SIZE = 500

@pytest.fixture()
def cluster_model_by_label() -> cluster_estimation_by_label.ClusterEstimationByLabel:  # type: ignore
    """
    Cluster estimation by label object.
    """
    result, test_logger = logger.Logger.create("test_logger", False)
    assert result
    assert test_logger is not None

    result, model = cluster_estimation_by_label.ClusterEstimationByLabel.create(
        MIN_TOTAL_POINTS_THRESHOLD,
        MIN_NEW_POINTS_TO_RUN,
        MAX_NUM_COMPONENTS,
        RNG_SEED,
        test_logger,
    )
    assert result
    assert model is not None

    yield model  # type: ignore

def generate_cluster_data(
    n_samples_per_cluster: "list[int]",
    cluster_standard_deviation: int,
    label: int,
) -> "tuple[list[detection_in_world.DetectionInWorld], list[np.ndarray]]":
    """
    Returns a list of points (DetectionInWorld objects) with specified points per cluster
    and standard deviation.

    PARAMETERS
    ----------
    n_samples_per_cluster: list[int]
        List corresponding to how many points to generate for each generated cluster
        ex: [10 20 30] will generate 10 points for one cluster, 20 points for the next,
        and 30 points for the final cluster.

    cluster_standard_deviation: int
        The standard deviation of the generated points, bigger
        standard deviation == more spread out points.

    label: int
        The label that every generated detection gets assigned

    RETURNS
    -------
    detections: list[detection_in_world.DetectionInWorld]
        List of points (DetectionInWorld objects).

    cluster_positions: list[np.ndarray]
        Coordinate positions of each cluster centre.
    -------
    """
    # .make_blobs() is a sklearn library function that returns a tuple of two values
    # First value is ndarray of shape (2, total # of samples) that gives the (x,y)
    # coordinate of generated data points.
    # Second value is the integer labels for cluster membership of each generated point (unused).
    # Third value is the (x,y) coordinates for each of the cluster centres.

    generated_points, _, cluster_positions = sklearn.datasets.make_blobs(  # type: ignore
        n_samples=n_samples_per_cluster,
        n_features=2,
        cluster_std=cluster_standard_deviation,
        center_box=(0, CENTRE_BOX_SIZE),
        random_state=RNG_SEED,
        return_centers=True,
    )

    detections = []
    for point in generated_points:
        # Placeholder variables to create DetectionInWorld objects
        placeholder_vertices = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        placeholder_confidence = 0.5

        result, detection_to_add = detection_in_world.DetectionInWorld.create(
            placeholder_vertices,
            point,
            label,
            placeholder_confidence,
        )

        assert result
        assert detection_to_add is not None
        detections.append(detection_to_add)

    return detections, cluster_positions.tolist()

def generate_cluster_data_by_label(
    labels_to_n_samples_per_cluster: "dict[int, list[int]]",
    cluster_standard_deviation: int,
) -> "tuple[list[detection_in_world.DetectionInWorld], dict[int, list[np.ndarray]]]":
    """
    Returns a list of labeled points (DetectionInWorld objects) with specified points per cluster
    and standard deviation.

    PARAMETERS
    ----------
    labels_to_cluster_samples: "dict[int, list[int]]"
        Dictionary where the key is a label and the value is a
        list of integers the represent the number of samples a cluster has.

    cluster_standard_deviation: int
        The standard deviation of the generated points, bigger
        standard deviation == more spread out points.

    RETURNS
    -------
    detections: list[detection_in_world.DetectionInWorld]
        List of points (DetectionInWorld objects).

    labels_to_cluster_positions: dict[int, list[np.ndarray]]
        Dictionary where the key is a label and the value is a
        list of coordinate positions of each cluster centre with that label.
    -------
    """

    detections = []
    labels_to_cluster_positions: dict[int, list[np.ndarray]] = {}

    for label, n_samples_list in labels_to_n_samples_per_cluster.items():
        temp_detections, cluster_positions = generate_cluster_data(
            n_samples_list, cluster_standard_deviation, label
        )
        detections += temp_detections
        labels_to_cluster_positions[label] = cluster_positions

    return detections, labels_to_cluster_positions

class TestModelExecutionCondition:
    """
    Tests execution condition for estimation worker at different amount of total and new data
    points.
    """

    __STD_DEV_REG = 1  # Regular standard deviation is 1m

    def test_under_min_total_threshold(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        Total data under threshold should not run.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Less than min threshold (100)

        generated_detections, _ = generate_cluster_data_by_label({0: [original_count]}, self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert not result
        assert detections_in_world is None

    def test_at_min_total_threshold(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        Should run once total threshold reached regardless of
        current bucket size.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Should not run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points

        generated_detections, _ = generate_cluster_data_by_label({0: [original_count]}, self.__STD_DEV_REG)
        generated_detections_2, _ = generate_cluster_data_by_label({0: [new_count]}, self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model_by_label.run(generated_detections_2, False)

        # Test
        assert not result
        assert detections_in_world is None
        assert result_2
        assert detections_in_world_2 is not None

    def test_under_min_bucket_size(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        New data under threshold should not run.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 10  # Should run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points, shouldn't run

        generated_detections, _ = generate_cluster_data_by_label({0: [original_count]}, self.__STD_DEV_REG)
        generated_detections_2, _ = generate_cluster_data_by_label({0: [new_count]}, self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model_by_label.run(generated_detections_2, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert not result_2
        assert detections_in_world_2 is None

    def test_good_data(self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel) -> None:
        """
        All conditions met should run.
        """
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 1  # More than min total threshold should run
        generated_detections, _ = generate_cluster_data_by_label({0: [original_count]}, self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None

class TestCorrectClusterPositionOutput:
    """
    Tests if cluster estimation by label properly sorts labels.
    """

    __STD_DEV_REG = 1  # Regular standard deviation is 1m
    __MAX_POSITION_TOLERANCE = 1

    def test_one_label(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        Five clusters with small standard devition that all have the same label
        """
        # Setup
        labels_to_n_samples_per_cluster = {1: [50, 100, 150, 200, 250]}
        generated_detections, labels_to_generated_cluster_positions = (
            generate_cluster_data_by_label(labels_to_n_samples_per_cluster, self.__STD_DEV_REG)
        )
        random.shuffle(
            generated_detections
        )  # so all abojects with the same label are not arranged all in a row

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world[1]) == 5
        for cluster in detections_in_world[1]:
            is_match = False
            for generated_cluster in labels_to_generated_cluster_positions[1]:
                # Check if coordinates are equal
                distance = np.linalg.norm(
                    [
                        cluster.location_x - generated_cluster[0],
                        cluster.location_y - generated_cluster[1],
                    ]
                )
                if distance < self.__MAX_POSITION_TOLERANCE:
                    is_match = True
                    break

            assert is_match

    def test_multiple_labels(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        Five clusters with small standard devition each belonging to one of three labels, with large points per cluster
        """
        # Setup
        labels_to_n_samples_per_cluster = {
            1: [70, 100, 130],
            2: [60, 90, 120],
            3: [50, 80, 110],
        }
        generated_detections, labels_to_generated_cluster_positions = (
            generate_cluster_data_by_label(labels_to_n_samples_per_cluster, self.__STD_DEV_REG)
        )
        random.shuffle(
            generated_detections
        )  # so all abojects with the same label are not arranged all in a row

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world[1]) == 3
        assert len(detections_in_world[2]) == 3
        assert len(detections_in_world[3]) == 3
        for label in range(1, 4):
            for cluster in detections_in_world[label]:
                is_match = False
                for generated_cluster in labels_to_generated_cluster_positions[label]:
                    # Check if coordinates are equal
                    distance = np.linalg.norm(
                        [
                            cluster.location_x - generated_cluster[0],
                            cluster.location_y - generated_cluster[1],
                        ]
                    )
                    if distance < self.__MAX_POSITION_TOLERANCE:
                        is_match = True
                        break

                assert is_match