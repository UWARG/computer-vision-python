import sklearn 
import sklearn.datasets
import numpy as np
import sklearn.mixture
import matplotlib.pyplot as plt
import scipy
import funcs

no_components = 8;

co_type = "spherical"
X, y = sklearn.datasets.make_blobs(n_samples=150, centers=4, n_features=2, cluster_std=0.1, center_box=(0,500))

bgm = sklearn.mixture.BayesianGaussianMixture(covariance_type = co_type, n_components=no_components, random_state=42, weight_concentration_prior=1000000, mean_precision_prior=0.1, init_params='k-means++').fit(X)
Z = bgm.predict(X)

bgm_means = bgm.means_


# Remove excess clusters - "The number of effective components is therefore smaller than n_components."
# As such, n_components is on the (very) liberal side - need remove unused clusters
i = 0
j = 0
while i < len(bgm_means):
    j = i+1
    while j < len(bgm_means):
        if funcs.get_distance(bgm_means[i,0],bgm_means[i,1],bgm_means[j,0],bgm_means[j,1]) < 0.25:
            bgm_means = np.delete(bgm_means, j, 0)
        j += 1
    i += 1

points_per_mean = [0]*len(bgm_means)
#Landing pad radius -> diameter = 32in -> rad = 16in = 0.4064m
lp_radius = 0.4064
i = 0
while i < len(X):
    near_cluster = False
    j = 0
    while j < len(bgm_means):
        # 0.75 because we don't want to keep a potential landing pad location on the basis of a bounding box being
        # on the very edge - want the drone in the centre of the pad, hence the smaller tolerance
        # Value subject to change
        if funcs.get_distance(X[i][0], X[i][1], bgm_means[j][0], bgm_means[j][1]) < lp_radius * 0.75:
            points_per_mean[j] += 1
            near_cluster = True
        # Keep the ones within 3x radius though (subject to change), since these might be considered during subsequent
        # VGMM runs
        elif funcs.get_distance(X[i][0], X[i][1], bgm_means[j][0], bgm_means[j][1]) < lp_radius * 3:
            near_cluster = True
        j += 1
    # Remove outliers not near detected cluster (3x radius - keep all within these)
    if not near_cluster:
        X = np.delete(X, i, 0)
        Z = np.delete(Z, i, 0)
    i += 1

#Remove all detected clusters with no bounding boxes within 0.75m - these are not good enough to ensure
# drone does not land on edge 
i = 0
while i < len(bgm_means):
    if points_per_mean[i] == 0:
        bgm_means = np.delete(bgm_means, i, 0)
        points_per_mean.pop(i)
        i -= 1
    i += 1

# -1, -1 if no landing spots found
optimal_point = [-1, -1]
if(len(bgm_means) > 0):
    optimal_point = bgm_means[np.argmax(points_per_mean)]

print(optimal_point)

colors = ["red", "blue", "green", "orange", "darkblue", "purple", "black"]
for k, col in enumerate(colors):
    cluster_data = Z == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=20)
plt.scatter(bgm_means[:,0], bgm_means[:,1], c="black", marker='*')
plt.title("umu")
plt.show()

