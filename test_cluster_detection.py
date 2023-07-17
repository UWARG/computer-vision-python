from timeit import default_timer as timer
import sklearn.datasets
from modules.cluster_estimation.cluster_estimation import ClusterEstimation
from modules.cluster_estimation.detection_in_world import DetectionInWorld
import numpy as np
import pytest

MIN_TOTAL_POINTS_THRESHOLD = 100
MIN_NEW_POINTS_THRESHOLD = 10
CENTER_BOX_SIZE = 500

# @pytest.fixture()
def cluster_model():
    model = ClusterEstimation()
    return model

# @pytest.fixture()
def cluster_data(n_samples:list, cluster_standard_deviation=1):
    X, y = sklearn.datasets.make_blobs(n_samples=n_samples, 
                                       n_features=2, 
                                       cluster_std=cluster_standard_deviation, 
                                       center_box=(0,CENTER_BOX_SIZE), 
                                       random_state=0)
    
    detections = []
    for pair in X:
        detections.append(DetectionInWorld(pair[0], pair[1]))
    return detections, y


def test_no_run_too_few_total_points():
    """
    Total data under threshold should not run 
    """
    # Setup
    num_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1  # less than min threshold (100)
    X, y = cluster_data([num_data_points])
    model = cluster_model()

    # Run
    model_ran, detections_in_world = model.run(X, False)
    
    # Test
    assert(model_ran == False)
    assert(detections_in_world == None)

def test_no_run_too_few_new_points():
    """
    New data under threshold should not run 
    """
    # Setup
    num_data_points = MIN_TOTAL_POINTS_THRESHOLD + 10  # should run the first time
    few_data_points = MIN_NEW_POINTS_THRESHOLD - 1 # under 10 new points 

    X, y = cluster_data([num_data_points])
    model = cluster_model()
    model.run(X, False)
    X, y = cluster_data([few_data_points])

    # Run
    model_ran, detections_in_world = model.run(X, False)
    
    # Test
    assert(model_ran == False)
    assert(detections_in_world == None)
    
def test_do_run_once_threshold_reached():
    """
    Runs at least once when min total points threshold reached regardless of new bucket size 
    """
    # Setup
    num_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1  # should not run the first time
    few_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1 # under min new points 

    X, y = cluster_data([num_data_points])
    model = cluster_model()
    model.run(X, False)
    X, y = cluster_data([few_data_points])

    # Run
    model_ran, detections_in_world = model.run(X, False)
    
    # Test
    assert(model_ran == True)
    assert(detections_in_world != None)

def test_do_run_regular_data():
    """
    Total data under threshold should not run 
    """
    # Setup
    num_data_points = MIN_TOTAL_POINTS_THRESHOLD + 1  # more than min threshold should run 
    X, y = cluster_data([num_data_points])
    model = cluster_model()

    # Run
    model_ran, detections_in_world = model.run(X, False)
    
    # Test
    assert(model_ran == True)
    assert(detections_in_world != None)

def test_detect_correct_amount_clusters():
    """
    Model detects correct number of clusters according to input data.
    Input 2 - 10 actual clusters
    Average standard deviation, same for all cluster: CENTER_BOX_SIZE / 100
    """

    # Setup
    MAX_NUM_CLUSTERS = 10

    data_generator_input_list = []  # create list for blob generator corresponding to 1 -> 10 clusters
    for i in range(MAX_NUM_CLUSTERS):
        data_generator_input_list.append([])
        for k in range(i+1):
            data_generator_input_list[i].append(MIN_TOTAL_POINTS_THRESHOLD+1)
    
    # Run
    model_runs:list[bool] = []
    list_detections_in_world:list[list[DetectionInWorld]] = []

    for data in data_generator_input_list:
        # Generate data & run model 
        X, y = cluster_data(data)
        model = cluster_model()
        run_status, detections_in_world = model.run(X, False)
        # Store results 
        model_runs.append(run_status)
        list_detections_in_world.append(detections_in_world)
    
    # Test
    assert(all(model_runs) == True)
    for i in range(MAX_NUM_CLUSTERS):
        print(f"{i} number of real clusters: {len(detections_in_world[i])} number of detected clusters")
        assert(len(detections_in_world[i]) == i)

# def test_cluster_estimation_worker_speed(): 
#     total = 0
#     no_iterations = 2

#     object_in_world_list = []

#     for i in range(no_iterations):
#         simulated_detections, y = sklearn.datasets.make_blobs(n_samples=[200,100, 30, 1], n_features=2, cluster_std=0.1, center_box=(0,500), random_state=0)
#         start = timer()
#         # Debug
#         clusest = ClusterEstimation()
#         ret_bool, ret_list = clusest.run(np.ndarray.tolist(simulated_detections), False)
#         # print(ret_bool)
#         # print(len(ret_list))
#         # for i in ret_list:
#         # print("x: %f, y: %f, var: %f" % (i.position_x, i.position_y, i.spherical_variance))
#         object_in_world_list += ret_list
#         end = timer()
#         total += (end - start)

#     print(len(object_in_world_list))
#     for cluster_center in object_in_world_list:
#         print()
#         print(cluster_center.position_x)
#         print(cluster_center.position_y)
#         print(cluster_center.spherical_variance)

#     print("Average execution time:")
#     print(total/no_iterations)