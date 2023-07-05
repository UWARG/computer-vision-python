
from modules.cluster_estimation.cluster_estimation import ClusterEstimation
import sklearn.datasets
import numpy as np


# Debug
if __name__ == "__main__":
    simulated_detections, y = sklearn.datasets.make_blobs(n_samples=150, centers=4, n_features=2, cluster_std=0.1, center_box=(0,500))
    clusest = ClusterEstimation()
    ret_bool, ret_list = clusest.run(np.ndarray.tolist(simulated_detections), False)
    print(ret_bool)
    print(len(ret_list))
    for i in ret_list:
        print("x: %f, y: %f, var: %f" % (i.position_x, i.position_y, i.spherical_variance))