# importing dependencies
import sys

import numpy as np

# function to plot the selected centroids
from scrips.Plot import plot_k_means


# function to compute euclidean distance
def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


# initialization algorithm
def initialize(data, n, plot=False):
    """
    initialized the centroids for K-means++
    inputs:
             data - numpy array of data points having shape (200, 2)
             k - number of clusters
    """
    # initialize the centroids list and add
    # a randomly selected data point to the list
    centroids = [data[np.random.randint(data.shape[0]), :]]

    # compute remaining k - 1 centroids
    for c_id in range(n - 1):

        # initialize a list to store distances of data
        # points from nearest centroid
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            # compute distance of 'point' from each of the previously
            # selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # select data point with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    if plot:
        plot_k_means(data, np.array(centroids))
    return np.array(centroids)
