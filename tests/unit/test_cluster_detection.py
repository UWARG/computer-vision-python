"""
Testing ClusterEstimation.
"""

import random
import numpy as np
import pytest
import sklearn.datasets

from modules.cluster_estimation import cluster_estimation
from modules.cluster_estimation import cluster_estimation_by_label
from modules.common.modules.logger import logger
from modules import detection_in_world


MIN_TOTAL_POINTS_THRESHOLD = 100
MIN_NEW_POINTS_TO_RUN = 10
MAX_NUM_COMPONENTS = 10
RNG_SEED = 0
CENTRE_BOX_SIZE = 500

# Test functions use test fixture signature names and access class privates
# No enable
# pylint: disable=protected-access,redefined-outer-name


@pytest.fixture()
def cluster_model() -> cluster_estimation.ClusterEstimation:  # type: ignore
    """
    Cluster estimation object.
    """
    result, test_logger = logger.Logger.create("test_logger", False)
    assert result
    assert test_logger is not None

    result, model = cluster_estimation.ClusterEstimation.create(
        MIN_TOTAL_POINTS_THRESHOLD,
        MIN_NEW_POINTS_TO_RUN,
        MAX_NUM_COMPONENTS,
        RNG_SEED,
        test_logger,
        0
    )
    assert result
    assert model is not None

    yield model  # type: ignore


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
        RNG_SEED,
        test_logger,
    )
    assert result
    assert model is not None

    yield model  # type: ignore


@pytest.fixture()
def cluster_model_by_label(cluster_model: cluster_estimation.ClusterEstimation) -> cluster_estimation_by_label.ClusterEstimationByLabel:  # type: ignore
    """
    Cluster estimation by label object.
    """
    result, test_logger = logger.Logger.create("test_logger", False)
    assert result
    assert test_logger is not None

    result, model = cluster_estimation_by_label.ClusterEstimationByLabel.create(
        MIN_TOTAL_POINTS_THRESHOLD,
        MIN_NEW_POINTS_TO_RUN,
        cluster_model,
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


def generate_points_away_from_cluster(
    num_points_to_generate: int,
    minimum_distance_from_cluster: float,
    cluster_positions: "list[np.ndarray]",
) -> "list[detection_in_world.DetectionInWorld]":
    """
    Returns a list of points with each point being the specified distance away from input cluster
    centre positions.

    PARAMETERS
    ----------
    num_points_to_generate: int
        Number of points to generate.

    minimum_distance_from_cluster: float
        Distance each generated point has to be from cluster centre.

    cluster_positions: list[np.ndarray]
        List containing cluster centres for points to be generated away from.

    RETURNS
    -------
    points_to_return: list[detection_in_world.DetectionInWorld]
        List of DetectionInWorld objects corresponding to points generated by function.
    """
    detections = []

    # Initialize random generator
    rng = np.random.default_rng(seed=RNG_SEED)

    while len(detections) < num_points_to_generate:
        # Generate random point
        point = rng.uniform(0, CENTRE_BOX_SIZE, size=2)
        valid = True

        # Check if outside minimum distance to cluster centres
        for cluster in cluster_positions:
            if np.linalg.norm(point - cluster) < minimum_distance_from_cluster:
                valid = False
                break

        if not valid:
            continue

        # Placeholder variables to create DetectionInWorld objects
        placeholder_vertices = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        placeholder_label = 1
        placeholder_confidence = 0.5

        result, detection_to_add = detection_in_world.DetectionInWorld.create(
            placeholder_vertices,
            point,
            placeholder_label,
            placeholder_confidence,
        )

        # Check that DetectionInWorld object created correctly
        assert result
        assert detection_to_add is not None

        # Add to list for return
        detections.append(detection_to_add)

    return detections


class TestModelExecutionCondition:
    """
    Tests execution condition for estimation worker at different amount of total and new data
    points.
    """

    __STD_DEV_REG = 1  # Regular standard deviation is 1m

    def test_under_min_total_threshold(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Total data under threshold should not run.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Less than min threshold (100)
        generated_detections, _ = generate_cluster_data([original_count], self.__STD_DEV_REG, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert not result
        assert detections_in_world is None

    def test_under_min_total_threshold_by_label(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        As above, but with labels.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Less than min threshold (100)
        generated_detections, _ = generate_cluster_data_by_label([1], [original_count], self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert not result
        assert detections_in_world is None


    def test_at_min_total_threshold(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Should run once total threshold reached regardless of
        current bucket size.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Should not run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points

        generated_detections, _ = generate_cluster_data([original_count], self.__STD_DEV_REG, 0)
        generated_detections_2, _ = generate_cluster_data([new_count], self.__STD_DEV_REG, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model.run(generated_detections_2, False)

        # Test
        assert not result
        assert detections_in_world is None
        assert result_2
        assert detections_in_world_2 is not None

    def test_at_min_total_threshold_by_label(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        As above, but with labels.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD - 1  # Should not run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points

        generated_detections, _ = generate_cluster_data_by_label([1], [original_count], self.__STD_DEV_REG)
        generated_detections_2, _ = generate_cluster_data_by_label([1], [new_count], self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model_by_label.run(generated_detections_2, False)

        # Test
        assert not result
        assert detections_in_world is None
        assert result_2
        assert detections_in_world_2 is not None


    def test_under_min_bucket_size(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        New data under threshold should not run.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 10  # Should run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points, shouldn't run

        generated_detections, _ = generate_cluster_data([original_count], self.__STD_DEV_REG, 0)
        generated_detections_2, _ = generate_cluster_data([new_count], self.__STD_DEV_REG, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model.run(generated_detections_2, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert not result_2
        assert detections_in_world_2 is None

    def test_under_min_bucket_size_by_label(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        As above, but with labels.
        """
        # Setup
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 10  # Should run the first time
        new_count = MIN_NEW_POINTS_TO_RUN - 1  # Under 10 new points, shouldn't run

        generated_detections, _ = generate_cluster_data_by_label([1], [original_count], self.__STD_DEV_REG)
        generated_detections_2, _ = generate_cluster_data_by_label([1], [new_count], self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)
        result_2, detections_in_world_2 = cluster_model_by_label.run(generated_detections_2, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert not result_2
        assert detections_in_world_2 is None
    

    def test_good_data(self, cluster_model: cluster_estimation.ClusterEstimation) -> None:
        """
        All conditions met should run.
        """
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 1  # More than min total threshold should run
        generated_detections, _ = generate_cluster_data([original_count], self.__STD_DEV_REG, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
    
    def test_good_data_by_label(self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel) -> None:
        """
        As above, but with labels.
        """
        original_count = MIN_TOTAL_POINTS_THRESHOLD + 1  # More than min total threshold should run
        generated_detections, _ = generate_cluster_data_by_label([1], [original_count], self.__STD_DEV_REG)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None


class TestCorrectNumberClusterOutputs:
    """
    Tests if cluster estimation output matches number of input clusters
    at different data scenarios.
    """

    __STD_DEV_REGULAR = 1
    __STD_DEV_LARGE = 5

    def test_detect_normal_data_single_cluster(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with small distribution and equal number of points per cluster centre.
        """
        # Setup
        points_per_cluster = [100]
        generated_detections, _ = generate_cluster_data(
            points_per_cluster, self.__STD_DEV_REGULAR, 0
        )

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None

    def test_detect_normal_data_single_cluster_by_label(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        As above, but with labels.
        """
        # Setup
        points_per_cluster = [100]
        generated_detections, _ = generate_cluster_data_by_label([1], points_per_cluster, self.__STD_DEV_REGULAR)

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None


    def test_detect_normal_data_five_clusters(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with small distribution and equal number of points per cluster centre.
        """
        # Setup
        points_per_cluster = [100, 100, 100, 100, 100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(
            points_per_cluster, self.__STD_DEV_REGULAR, 0
        )

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count

    def test_detect_normal_data_five_clusters_by_label_all_different(
        self, cluster_model_by_label: cluster_estimation_by_label.ClusterEstimationByLabel
    ) -> None:
        """
        As above, but with labels. Every cluster has a different label.
        """
        # Setup
        points_per_cluster = [100, 100, 100, 100, 100]
        labels_of_clusters = [1, 1, 1, 1, 1]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, clusters = generate_cluster_data_by_label(labels_of_clusters, points_per_cluster, self.__STD_DEV_REGULAR)
        assert len(generated_detections) == 500
        assert len(clusters) == 5

        # Run
        result, detections_in_world = cluster_model_by_label.run(generated_detections, False)

        # Test
        assert detections_in_world[0].label == 1
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count
    

    def test_detect_large_std_dev_single_cluster(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with large distribution and equal number of points per cluster centre.
        """
        # Setup
        points_per_cluster = [100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(points_per_cluster, self.__STD_DEV_LARGE, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count

    def test_detect_large_std_dev_five_clusters(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with large distribution and equal number of points per cluster centre.
        """
        # Setup
        points_per_cluster = [100, 100, 100, 100, 100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(points_per_cluster, self.__STD_DEV_LARGE, 0)

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count

    def test_detect_skewed_data_single_cluster(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with small distribution but varying number of points per cluster centre and
        random outlier points to simulate false detections.
        """
        # Setup
        points_per_cluster = [10, 100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(
            points_per_cluster, self.__STD_DEV_REGULAR, 0
        )

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count

    def test_detect_skewed_data_five_clusters(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Data with small distribution but varying number of points per cluster centre and
        random outlier points to simulate false detections.
        """
        # Setup
        points_per_cluster = [20, 100, 100, 100, 100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, cluster_positions = generate_cluster_data(
            points_per_cluster,
            self.__STD_DEV_REGULAR,
            0,
        )

        # Add 5 random points to dataset, each being at least 20m away from cluster centres
        outlier_count = 5
        outlier_distance_from_cluster_centre = 20
        outlier_detections = generate_points_away_from_cluster(
            outlier_count,
            outlier_distance_from_cluster_centre,
            cluster_positions,
        )
        generated_detections += outlier_detections

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None
        assert len(detections_in_world) == expected_cluster_count

    def test_detect_consecutive_inputs_single_cluster(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Previous tests executed model with all points available at once. This test feeds the model
        one point from the dataset one at a time (and calls .run() each time), checking for correct
        number of output clusters once all points have been inputted.
        """
        # Setup
        points_per_cluster = [100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(
            points_per_cluster, self.__STD_DEV_REGULAR, 0
        )

        # Run
        result_latest = False
        detections_in_world_latest = []
        for point in generated_detections:
            result_latest, detections_in_world_latest = cluster_model.run([point], False)

        # Test
        assert result_latest
        assert detections_in_world_latest is not None
        assert len(detections_in_world_latest) == expected_cluster_count

    def test_detect_consecutive_inputs_five_clusters(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Previous tests executed model with all points available at once. This test feeds the model
        one point from the dataset one at a time (and calls .run() each time), checking for correct
        number of output clusters once all points have been inputted.
        """
        # Setup
        points_per_cluster = [100, 100, 100, 100, 100]
        expected_cluster_count = len(points_per_cluster)
        generated_detections, _ = generate_cluster_data(
            points_per_cluster, self.__STD_DEV_REGULAR, 0
        )

        # Run
        result_latest = False
        detections_in_world_latest = []
        for point in generated_detections:
            result_latest, detections_in_world_latest = cluster_model.run([point], False)

        # Test
        assert result_latest
        assert detections_in_world_latest is not None
        assert len(detections_in_world_latest) == expected_cluster_count


class TestCorrectClusterPositionOutput:
    """
    Tests if cluster estimation output falls within acceptable distance to
    input cluster positions.
    """

    __STD_DEV_REG = 1  # Regular standard deviation is 1m
    __MAX_POSITION_TOLERANCE = 1

    def test_position_regular_data(
        self, cluster_model: cluster_estimation.ClusterEstimation
    ) -> None:
        """
        Five clusters with small standard deviation and large number of points per cluster.
        """
        # Setup
        points_per_cluster = [100, 100, 100, 100, 100]
        generated_detections, cluster_positions = generate_cluster_data(
            points_per_cluster,
            self.__STD_DEV_REG,
            0,
        )

        # Run
        result, detections_in_world = cluster_model.run(generated_detections, False)

        # Test
        assert result
        assert detections_in_world is not None

        # Check if within acceptable distance
        for detection in detections_in_world:
            is_match = False
            for position in cluster_positions:
                # Get distance between predicted cluster and actual cluster
                distance = np.linalg.norm(
                    [detection.location_x - position[0], detection.location_y - position[1]]
                )

                # Check tolerance
                if distance < self.__MAX_POSITION_TOLERANCE:
                    is_match = True
                    break

            assert is_match


class TestCorrectClusterEstimationByLabel:
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
        labels_to_n_samples_per_cluster = {1: [100, 100, 100, 100, 100]}
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
            assert cluster.label == 1
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
        Five clusters with small standard devition that have different labels
        """
        # Setup
        labels_to_n_samples_per_cluster = {
            1: [100, 100, 100],
            2: [100, 100, 100],
            3: [100, 100, 100],
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
                assert cluster.label == label
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
