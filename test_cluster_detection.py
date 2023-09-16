import pytest
import numpy as np
import sklearn.datasets


from modules.cluster_estimation.cluster_estimation import ClusterEstimation
from modules.detection_in_world import DetectionInWorld


MIN_TOTAL_POINTS_THRESHOLD = 100
MIN_NEW_POINTS_TO_RUN = 10
CENTER_BOX_SIZE = 500
RANDOM_STATE = 0

@pytest.fixture()
def cluster_model():
    model_created, model = ClusterEstimation.create(MIN_NEW_POINTS_TO_RUN, MIN_NEW_POINTS_TO_RUN, RANDOM_STATE)
    assert model_created
    yield model

def generate_cluster_data(n_samples:list[int], cluster_standard_deviation:int=1) -> list[DetectionInWorld]:
    """
    Returns a list of points (DetectionInWorld objects) with specified points per cluster
    and standard deviation 

    PARAMETERS
    ----------
    n_samples: list[int]
        list corresponding to how many points to generate for each generated cluster
        ex: [10 20 30] will generate 10 points for one cluster, 20 points for the next, 
        and 30 points for the final cluster

    cluster_standard_deviation: int
        the standard deviation of the generated points, bigger
        standard deviation == more spread out points

    RETURNS
    -------
    """
    # .make_blobs() is a sklearn library function that returns a tuple of two values
    # first value is ndarray of shape (2, total # of samples) that gives the (x,y) 
    # coordinate of generated data points
    # second value is the integer labels for cluster membership of each generated point.

    generated_points, labels, cluster_positions = sklearn.datasets.make_blobs(n_samples=n_samples, 
                                       n_features=2, 
                                       cluster_std=cluster_standard_deviation, 
                                       center_box=(0,CENTER_BOX_SIZE), 
                                       random_state=0,
                                       return_centers=True)
    detections = []
    for point in generated_points:
        # Placeholder variables to create DetectionInWorld objects
        placeholder_verticies = np.array([[0,0],[0,0],[0,0],[0,0]])
        placeholder_label = 1
        placeholder_confidence = 0.5

        detection_created, detection_to_add = DetectionInWorld.create(placeholder_verticies,
                                                                      point,
                                                                      placeholder_label,
                                                                      placeholder_confidence)
        
        if (detection_created):
            detections.append(detection_to_add)
            
    return detections, labels, cluster_positions.tolist()

class TestModelExecutionCondition():
    """
    Tests execution condition for estimation worker at different amount of total and new data points
    """

    def test_under_min_total_threshold(self, cluster_model: ClusterEstimation):
        """
        Total data under threshold should not run 
        """
        # Setup 
        TOTAL_NUM_DETECTIONS = MIN_TOTAL_POINTS_THRESHOLD - 1  # less than min threshold (100)
        generated_detections, labels, cluster_positions = generate_cluster_data([TOTAL_NUM_DETECTIONS])

        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(not model_ran)
        assert(detections_in_world is None)
    
    def test_at_min_total_threshold(self, cluster_model:ClusterEstimation):
        """
        Should run once once total threshold reached regardless of
        current bucket size
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1  # should not run the first time
        NEW_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1 # under 10 new points 
        
        generated_detections, labels, cluster_positions = generate_cluster_data([NUM_DATA_POINTS])
        generated_detections_2, labels_2, cluster_positions_2 = generate_cluster_data([NEW_DATA_POINTS])

        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        model_ran, detections_in_world = cluster_model.run(generated_detections_2, False)

        # Test
        assert(model_ran)
        assert(detections_in_world is not None)

    def test_under_min_bucket_size(self, cluster_model:ClusterEstimation):
        """
        New data under threshold should not run 
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD + 10  # should run the first time
        NEW_DATA_POINTS = MIN_NEW_POINTS_TO_RUN - 1 # under 10 new points 

        generated_detections, labels, cluster_positions= generate_cluster_data([NUM_DATA_POINTS])
        generated_detections_2, labels_2, cluster_positions_2 = generate_cluster_data([NEW_DATA_POINTS])

        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        model_ran, detections_in_world = cluster_model.run(generated_detections_2, False)

        # Test
        assert(not model_ran)
        assert(detections_in_world is None)

    def test_good_data(self, cluster_model:ClusterEstimation):
        """
        All conditions met should run
        """
        EXPECTED_CLUSTER_COUNT = MIN_TOTAL_POINTS_THRESHOLD + 1  # more than min total threshold should run 
        generated_detections, labels, cluster_positions= generate_cluster_data([EXPECTED_CLUSTER_COUNT])
    
        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(model_ran)
        assert(detections_in_world is not None)
    

class TestCorrectNumberClusterOutputs():
    """
    Tests if cluster estimation output matches number of input clusters
    at different data scenarios. 
    """
    def test_detect_normal_data(self, cluster_model:ClusterEstimation):
        """
        data with small distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [100, 100, 100, 100, 100]
        generated_detections, labels, cluster_positions= generate_cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        cluster_model.clear_all_data()
        cluster_model.reset_model()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(model_ran)
        assert(detections_in_world is not None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)

    def test_detect_large_std_dev(self, cluster_model:ClusterEstimation):
        """
        data with large distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_LARGE = CENTER_BOX_SIZE / 100
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [100, 100, 100, 100, 100]
        generated_detections, labels, cluster_positions= generate_cluster_data(DATA, STD_DEV_LARGE)

        # Run
        cluster_model.clear_all_data()
        cluster_model.reset_model()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(model_ran)
        assert(detections_in_world is not None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)

    def test_detect_skewed_data(self, cluster_model:ClusterEstimation):
        """
        data with small distribution but varying number of points per cluster center and
        random outlier points to simulate false detections
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [1, 1, 1, 1, 20, 100, 100, 100, 100]
        generated_detections, labels, cluster_positions = generate_cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        cluster_model.clear_all_data()
        cluster_model.reset_model()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(model_ran)
        assert(detections_in_world is not None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)

class TestCorrectCluterPositionOutput:
    """
    Tests if cluster estimation output falls within acceptable distance to
    input cluster positions
    """
    def test_position_regular_data(self, cluster_model:ClusterEstimation):
        """
        5 clusters with small standard deviation and large number of points per cluster
        """
        # setup
        MAX_POSITION_INACCURACY = 1  # accuracy within 1m
        std_dev_regular = CENTER_BOX_SIZE / 500
        expected_cluster_count = 5
        data = [100, 100, 100, 100, 100]
        generated_detections, labels, cluster_positions = generate_cluster_data(data, std_dev_regular)

        # run
        cluster_model.clear_all_data()
        cluster_model.reset_model()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)

        # check if within acceptable distance
        temp = []
        for detection in detections_in_world:
            for position in cluster_positions:
                if np.linalg.norm([detection.position_x - position[0], detection.position_y - position[1]]) < MAX_POSITION_INACCURACY:
                    temp.append(position)

        # test
        assert(model_ran)
        assert(detections_in_world is not None)
        assert(len(temp) == expected_cluster_count)

        return
