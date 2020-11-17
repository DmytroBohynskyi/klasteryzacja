import numpy as np
import skfuzzy as fuzz

from scrips.Plot import plot_k_means


class K_means:
    def __init__(self, data, classes_num, plot=False, data_type=None):
        self.data: np.ndarray = data  # numpy array of data points
        self.classes_num: int = classes_num
        self.centroids: np.ndarray = self.initialize(data, classes_num)  # position of centroids
        self.dimension = data.shape[1]
        self.save_path = None
        self.save_data = None
        self.data_type = data_type

    @staticmethod
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

            dist = K_means.sort_by_distance(data, centroids)

            # select data point with maximum distance as our next centroid
            dist = np.min(dist, axis=1)
            next_centroid = data[np.argmax(dist), :]
            centroids.append(next_centroid)

        if plot:
            plot_k_means(data, np.array(centroids))
        return np.array(centroids)

    def k_means(self, **kwargs):
        """
        start of segmentation ( K-means++)
        inputs:
                 data - numpy array of data points having shape (200, 2)
                 centroids - position of centroids
        """
        # variables for comparing segmentation results
        new_array, old_array = 1, 0
        # main loop, if new_array equal old_array stop loop
        while not np.array_equal(new_array, old_array):
            old_array = new_array

            # sort_by_distance, looking for distances to centroid
            new_array = self.sort_by_distance(self.data, self.centroids)
            # looking for the nearest centroid
            self.data_type = np.argmin(new_array, axis=1)

            self.clustering_empty()

        self.plot(label=f"K_means: cluster = {self.classes_num}") if kwargs.get('plot') else None
        self.save() if kwargs.get("save") else None

    @staticmethod
    def sort_by_distance(data, centroids):
        array = np.zeros([len(data), len(centroids)])
        for n_centroid, centroid in enumerate(centroids):
            array[:, n_centroid] = np.power(np.sum((data - centroid) ** 2, axis=1), 0.5)
        return array

    def clustering_empty(self):
        for n, centroid in enumerate(self.centroids):
            data_type_index = np.where(self.data_type == n)  # index value for n centroid
            if len(data_type_index[0]) != 0:  # next if n centroid dont have value
                self.centroids[n] = np.average(self.data[data_type_index], axis=0)

    def fuzzy(self):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(self.data.T, self.classes_num, self.dimension,
                                                         error=0.005, maxiter=1000, init=None)
        self.data_type = np.argmax(u, axis=0)
        self.centroids = cntr

        self.plot(label="Fuzzy_means")

    def plot(self, label="K_means"):
        plot_k_means(self.data, self.centroids, color=self.data_type, fig_type=label)

    def save(self):
        save_data = np.append(self.data.T, [self.data_type], axis=0).T
        self.save_data = save_data[save_data[:, 2].argsort()]  # sort by type
        self.save_path = f'models/K_means_{self.classes_num}_{self.dimension}.csv'
        np.savetxt(self.save_path, self.save_data, delimiter=',')
