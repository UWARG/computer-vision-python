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

def cluster_data(n_samples:list, cluster_standard_deviation:int=1) -> list[DetectionInWorld]:
    """
    Returns a list of points (DetectionInWorld objects) with specified points per cluster
    and standard deviation 
    """
    X, y = sklearn.datasets.make_blobs(n_samples=n_samples, 
                                       n_features=2, 
                                       cluster_std=cluster_standard_deviation, 
                                       center_box=(0,CENTER_BOX_SIZE), 
                                       random_state=0)
    detections = []
    for pair in X:
        detections.append(DetectionInWorld(pair[0], pair[1]))
    return detections, y

class Test_Model_Execution_Condition():
    """
    Tests execution condition for estimation worker at different amount of total and new data points
    """

    def test_under_min_total_threshold(self, cluster_model):
        """
        Total data under threshold should not run 
        """
        # Setup 
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1  # less than min threshold (100)
        X, y = cluster_data([NUM_DATA_POINTS])

        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(X, False)
        
        # Test
        assert(model_ran == False)
        assert(detections_in_world == None)
    
    def test_at_min_total_threshold(self, cluster_model):
        """
        Should run once once total threshold reached regardless of
        current bucket size
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1  # should not run the first time
        NEW_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD - 1 # under 10 new points 
        
        X, y = cluster_data([NUM_DATA_POINTS])
        X_2, y_2 = cluster_data([NEW_DATA_POINTS])

        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(X, False)
        model_ran, detections_in_world = model.run(X_2, False)

        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)

    def test_under_min_bucket_size(self, cluster_model):
        """
        New data under threshold should not run 
        """
        # Setup
        NUM_DATA_POINTS = MIN_TOTAL_POINTS_THRESHOLD + 10  # should run the first time
        NEW_DATA_POINTS = MIN_NEW_POINTS_THRESHOLD - 1 # under 10 new points 

        X, y = cluster_data([NUM_DATA_POINTS])
        X_2, y_2 = cluster_data([NEW_DATA_POINTS])

        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(X, False)
        model_ran, detections_in_world = model.run(X_2, False)

        # Test
        assert(model_ran == False)
        assert(detections_in_world == None)

    def test_good_data(self, cluster_model):
        """
        All conditions met should run
        """
        NUM_TRUE_CLUSTERS = MIN_TOTAL_POINTS_THRESHOLD + 1  # more than min total threshold should run 
        X, y = cluster_data([NUM_TRUE_CLUSTERS])
        model = cluster_model
    
        # Run
        model = cluster_model
        model.clear_all_data()
        model_ran, detections_in_world = model.run(X, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
    

class Test_Correct_Number_Cluster_Outputs():
    """
    Tests if cluster estimation output matches number of input clusters
    at different data scenarios. 
    """
    def test_detect_normal_data(self, cluster_model):
        """
        data with small distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        NUM_TRUE_CLUSTERS = 5
        DATA = [100, 100, 100, 100, 100]
        X, y = cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(X, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == NUM_TRUE_CLUSTERS)

    def test_detect_large_STDDEV(self, cluster_model):
        """
        data with large distribution and equal number of points per cluster center. 
        """
        # Setup
        STD_DEV_LARGE = CENTER_BOX_SIZE / 100
        NUM_TRUE_CLUSTERS = 5
        DATA = [100, 100, 100, 100, 100]
        X, y = cluster_data(DATA, STD_DEV_LARGE)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(X, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == NUM_TRUE_CLUSTERS)

    def test_detect_skewed_data(self, cluster_model):
        """
        data with small distribution but varying number of points per cluster center. 
        """
        # Setup
        STD_DEV_REGULAR = CENTER_BOX_SIZE / 500
        NUM_TRUE_CLUSTERS = 5
        DATA = [5, 100, 100, 100, 100]
        X, y = cluster_data(DATA, STD_DEV_REGULAR)

        # Run
        model = cluster_model
        model.clear_all_data()
        model.reset_model()
        model_ran, detections_in_world = model.run(X, False)
        
        # Test
        assert(model_ran == True)
        assert(detections_in_world != None)
        assert(len(detections_in_world) == NUM_TRUE_CLUSTERS)




# def test_filter_by_points_ownership():
#     """
#     Test filtering by points ownership
#     """
#     # Setup
#     weights = np.array([8.24927030e-01, 8.29623811e-02, 8.25596577e-02, 4.85827147e-03,
#        8.01693884e-04, 7.93756321e-04, 7.85897347e-04, 7.78116185e-04,
#        7.70412065e-04, 7.62784223e-04])
    
#     covariances = np.array([  2.65098783,  17.6815593 ,  17.58032972, 243.93659808,
#        851.98519218, 851.98519218, 851.98519218, 851.98519218,
#        851.98519218, 851.98519218])
    
#     clusters = np.array([[218.76667282, 445.88192477],
#        [274.36365232, 357.70294735],
#        [301.21728607, 272.34458143],
#        [211.86007248, 323.06428146],
#        [230.19424563, 423.65310983],
#        [230.19424563, 423.65310983],
#        [230.19424563, 423.65310983],
#        [230.19424563, 423.65310983],
#        [230.19424563, 423.65310983],
#        [230.19424563, 423.65310983]])
    
# def test_no_run_too_few_total_points(cluster_model):
#     """
#     Total data under threshold should not run 
#     """
#     # Setup
#     num_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1  # less than min threshold (100)
#     X, y = cluster_data([num_data_points])
#     model = cluster_model

#     # Run
#     model_ran, detections_in_world = model.run(X, False)
    
#     # Test
#     assert(model_ran == False)
#     assert(detections_in_world == None)

# def test_no_run_too_few_new_points(cluster_model):
#     """
#     New data under threshold should not run 
#     """
#     # Setup
#     num_data_points = MIN_TOTAL_POINTS_THRESHOLD + 10  # should run the first time
#     few_data_points = MIN_NEW_POINTS_THRESHOLD - 1 # under 10 new points 

#     X, y = cluster_data([num_data_points])
#     model = cluster_model
#     model.run(X, False)
#     X, y = cluster_data([few_data_points])

#     # Run
#     model_ran, detections_in_world = model.run(X, False)
    
#     # Test
#     assert(model_ran == False)
#     assert(detections_in_world == None)
    
# def test_do_run_once_threshold_reached(cluster_model):
#     """
#     Runs at least once when min total points threshold reached regardless of current bucket size 
#     """
#     # Setup
#     num_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1  # should not run the first time
#     few_data_points = MIN_TOTAL_POINTS_THRESHOLD - 1 # under min new points 

#     X, y = cluster_data([num_data_points])
#     model = cluster_model
#     model.run(X, False)
#     X, y = cluster_data([few_data_points])

#     # Run
#     model_ran, detections_in_world = model.run(X, False)
    
#     # Test
#     assert(model_ran == True)
#     assert(detections_in_world != None)

# def test_do_run_regular_data(cluster_model):
#     """
#     Total data over threshold should run normally
#     """
#     # Setup
#     num_data_points = MIN_TOTAL_POINTS_THRESHOLD + 1  # more than min threshold should run 
#     X, y = cluster_data([num_data_points])
#     model = cluster_model

#     # Run
#     model_ran, detections_in_world = model.run(X, False)
    
#     # Test
#     assert(model_ran == True)
#     assert(detections_in_world != None)

# def test_detect_correct_amount_clusters(cluster_model):
#     """
#     Model detects correct number of clusters according to input data.
#     Input 2 - 10 actual clusters
#     Average standard deviation, same for all cluster: CENTER_BOX_SIZE / 500
#     """

#     # Setup
#     MAX_NUM_CLUSTERS = 10

#     data_generator_input_list = []  # create list for blob generator corresponding to 1 -> 10 clusters
#     for i in range(MAX_NUM_CLUSTERS):
#         data_generator_input_list.append([])
#         for k in range(i+1):
#             data_generator_input_list[i].append(MIN_TOTAL_POINTS_THRESHOLD)
#     # Run
#     model_runs:list[bool] = []
#     list_detections_in_world:list[list[DetectionInWorld]] = []

#     for data in data_generator_input_list:
#         # Generate data & run model 
#         X, y = cluster_data(data)
#         model = cluster_model
#         #print(model)
#         run_status, detections_in_world = model.run(X, False)
#         # Store results 
#         model_runs.append(run_status)
#         list_detections_in_world.append(detections_in_world)
    
#     # Test
#     assert(all(model_runs) == True)
#     for i in range(MAX_NUM_CLUSTERS):
#         # print(f"{i+1} number of real clusters: {len(list_detections_in_world[i])} number of detected clusters")
#         # Same number of detections from model as cluster in world
#         assert(len(list_detections_in_world[i]) == i + 1)

# def test_detect_correct_amount_clusters_large_STDDEV(cluster_model):
#     """
#     2 - 10 actual clusters
#     Average standard deviation, same for all cluster: CENTER_BOX_SIZE / 100
#     """

#     # Setup
#     MAX_NUM_CLUSTERS = 10
#     STD_DEV = CENTER_BOX_SIZE / 100  # standard deviation ~5m (which is max size in real life hopefully)
    
#     data_generator_input_list = []  # create list for blob generator corresponding to 1 -> 10 clusters
#     for i in range(MAX_NUM_CLUSTERS):
#         data_generator_input_list.append([])
#         for k in range(i+1):
#             data_generator_input_list[i].append(MIN_TOTAL_POINTS_THRESHOLD)
#     # Run
#     model_runs:list[bool] = []
#     list_detections_in_world:list[list[DetectionInWorld]] = []

#     for data in data_generator_input_list:
#         # Generate data & run model
#         X, y = cluster_data(data, STD_DEV)
#         model = model = unique_cluster_model()
#         run_status, detections_in_world = model.run(X, False)
#         # Store results
#         model_runs.append(run_status)
#         list_detections_in_world.append(detections_in_world)
    
#     # Test
#     assert(all(model_runs) == True)
#     for i in range(MAX_NUM_CLUSTERS):
#         # print(f"{i+1} number of real clusters: {len(list_detections_in_world[i])} number of detected clusters")
#         # Same number of detections from model as cluster in world
#         assert(len(list_detections_in_world[i]) == i + 1)

# def test_skewed_data(cluster_model):
#     """
#     Model should detect correct number of clusters according to input data.
#     Input: 1 cluster with 5 points, rest of clusters have high number of points (101)
#     Average standard deviation, same for all cluster: CENTER_BOX_SIZE / 500
#     """

#     # Setup
#     MAX_NUM_CLUSTERS = 10
#     STD_DEV = CENTER_BOX_SIZE / 500  
    
#     data_generator_input_list = []  # create list for blob generator corresponding to 1 -> 10 clusters
#     for i in range(MAX_NUM_CLUSTERS):
#         data_generator_input_list.append([])
#         for k in range(i+1):
#             if k == 0:
#                 data_generator_input_list[i].append(5)
#             else:
#                 data_generator_input_list[i].append(MIN_TOTAL_POINTS_THRESHOLD)

#     # Run
#     model_runs:list[bool] = []
#     list_detections_in_world:list[list[DetectionInWorld]] = []

#     for data in data_generator_input_list:
#         # Generate data & run model 
#         X, y = cluster_data(data, STD_DEV)
#         model = model = unique_cluster_model()
#         run_status, detections_in_world = model.run(X, False)
#         # Store results 
#         model_runs.append(run_status)
#         list_detections_in_world.append(detections_in_world)
    
#     # Test
#     assert(all(model_runs[1:]) == True)
#     for i in range(MAX_NUM_CLUSTERS):
#         # print(f"{i+1} number of real clusters: {len(list_detections_in_world[i])} number of detected clusters")
#         # Same number of detections from model as cluster in world
#         if i != 0: 
#             assert(len(list_detections_in_world[i]) == i + 1)


# # def test_cluster_estimation_worker_speed(): 
# #     total = 0
# #     no_iterations = 2

# #     object_in_world_list = []

# #     for i in range(no_iterations):
# #         simulated_detections, y = sklearn.datasets.make_blobs(n_samples=[200,100, 30, 1], n_features=2, cluster_std=0.1, center_box=(0,500), random_state=0)
# #         start = timer()
# #         # Debug
# #         clusest = ClusterEstimation()
# #         ret_bool, ret_list = clusest.run(np.ndarray.tolist(simulated_detections), False)
# #         # print(ret_bool)
# #         # print(len(ret_list))
# #         # for i in ret_list:
# #         # print("x: %f, y: %f, var: %f" % (i.position_x, i.position_y, i.spherical_variance))
# #         object_in_world_list += ret_list
# #         end = timer()
# #         total += (end - start)

# #     print(len(object_in_world_list))
# #     for cluster_center in object_in_world_list:
# #         print()
# #         print(cluster_center.position_x)
# #         print(cluster_center.position_y)
# #         print(cluster_center.spherical_variance)

# #     print("Average execution time:")
# #     print(total/no_iterations)