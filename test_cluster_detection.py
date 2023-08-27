from timeit import default_timer as timer
import sklearn.datasets
from modules.cluster_estimation.cluster_estimation import ClusterEstimation
from modules.cluster_estimation.detection_in_world import DetectionInWorld
import numpy as np
import pytest

MIN_TOTAL_POINTS_THRESHOLD = 100
MIN_NEW_POINTS_THRESHOLD = 10
CENTER_BOX_SIZE = 500

@pytest.fixture()
def cluster_model():
    model = ClusterEstimation()
    yield model

def cluster_data(n_samples:list[int], cluster_standard_deviation:int=1) -> list[DetectionInWorld]:
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

    generated_points, labels = sklearn.datasets.make_blobs(n_samples=n_samples, 
                                       n_features=2, 
                                       cluster_std=cluster_standard_deviation, 
                                       center_box=(0,CENTER_BOX_SIZE), 
                                       random_state=0)
    detections = []
    for point in generated_points:
        detections.append(DetectionInWorld(point[0], point[1]))
    return detections, labels

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
        generated_detections, labels = cluster_data([TOTAL_NUM_DETECTIONS])

        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        
        # Test
        assert(not model_ran)
        assert(detections_in_world is None)
    
    def test_at_min_total_threshold(self, cluster_model: ClusterEstimation):
        """
        Should run once once total threshold reached regardless of
        current bucket size
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1  # should not run the first time
        NEW_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1 # under 10 new points 
        
        generated_detections, y = cluster_data([NUM_DATA_POINTS])
        generated_detections_2, y_2 = cluster_data([NEW_DATA_POINTS])

        # Run
        cluster_model.clear_all_data()
        model_ran, detections_in_world = cluster_model.run(generated_detections, False)
        model_ran, detections_in_world = cluster_model.run(generated_detections_2, False)

        # Test
        assert(model_ran)
        assert(detections_in_world is not None)

    def test_under_min_bucket_size(self, cluster_model: ClusterEstimation):
        """
        New data under threshold should not run 
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD + 10  # should run the first time
        NEW_DATA_POINTS = MIN_NEW_POINTS_THRESHOLD - 1 # under 10 new points 

        generated_detections, y = cluster_data([NUM_DATA_POINTS])
        generated_detections_2, y_2 = cluster_data([NEW_DATA_POINTS])

        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(generated_detections, False)
        model_ran, detections_in_world = model.run(generated_detections_2, False)

        # Test
        assert(not model_ran)
        assert(detections_in_world is None)

    def test_good_data(self, cluster_model: ClusterEstimation):
        """
        All conditions met should run
        """
        EXPECTED_CLUSTER_COUNT = MIN_TOTAL_POINTS_THRESHOLD + 1  # more than min total threshold should run 
        generated_detections, y = cluster_data([EXPECTED_CLUSTER_COUNT])
        model = cluster_model
    
        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(generated_detections, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
    

class TestCorrectNumberClusterOutputs():
    """
    Tests if cluster estimation output matches number of input clusters
    at different data scenarios. 
    """
    def test_detect_normal_data(self, cluster_model: ClusterEstimation):
        """
        data with small distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [100, 100, 100, 100, 100]
        generated_detections, y = cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(generated_detections, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)

    def test_detect_large_std_dev(self, cluster_model: ClusterEstimation):
        """
        data with large distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_LARGE = CENTER_BOX_SIZE / 100
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [100, 100, 100, 100, 100]
        generated_detections, y = cluster_data(DATA, STD_DEV_LARGE)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(generated_detections, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)

    def test_detect_skewed_data(self, cluster_model: ClusterEstimation):
        """
        data with small distribution but varying number of points per cluster center. 
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        EXPECTED_CLUSTER_COUNT = 5
        DATA = [5, 100, 100, 100, 100]
        generated_detections, y = cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(generated_detections, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == EXPECTED_CLUSTER_COUNT)