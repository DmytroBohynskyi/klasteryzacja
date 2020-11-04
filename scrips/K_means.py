import numpy as np
from numpy import savetxt

from scrips.Plot import plot_k_means

COLOR = ["#ffce00", "#5f720f", "#6c235f", "#69173a", "#0a70cf", "#89ff33", "#2f82d5", "#d47332", "#a8255d", "#653244",
         "#7a1717", "#336f3b", "#006de1", "#1f0a1b", "#751a41", ]


class K_means:

    def __init__(self, ):

        self.data = None  # numpy array of data points
        self.centroids = None  # position of centroids
        self.classes_num = None
        self.dimension = None
        self.save_path = None
        self.save_data = None
        self.data_type = None

    def start(self, data: np.ndarray, centroids: np.ndarray, **kwargs):
        """
        start of segmentation ( K-means++)
        inputs:
                 data - numpy array of data points having shape (200, 2)
                 centroids - position of centroids
        """
        self.data: np.ndarray = data
        self.centroids: np.ndarray = centroids
        self.classes_num = centroids.shape[0]
        self.dimension = data.shape[1]

        self.start_algorithm()
        self.plot() if kwargs.get('plot') else None
        self.save() if kwargs.get("save") else None

    def start_algorithm(self):
        # variables for comparing segmentation results
        new_array, old_array = 1, 0
        # main loop, if new_array equal old_array stop loop
        while not np.array_equal(new_array, old_array):
            old_array = new_array

            # sort_by_distance, looking for distances to centroid
            new_array = self.sort_by_distance()
            # looking for the nearest centroid
            data_type = np.argmin(new_array, axis=1)

            self.clustering_empty(data_type)
        self.data_type = data_type
        self.save_data = np.append(self.data.T, [data_type], axis=0).T

    def sort_by_distance(self):
        array = np.zeros([len(self.data), len(self.centroids)])
        for n_centroid, centroid in enumerate(self.centroids):
            array[:, n_centroid] = np.power(np.sum((self.data - centroid) ** 2, axis=1), 0.5)
        return array

    def clustering_empty(self, data_type: np.ndarray):
        for n, centroid in enumerate(self.centroids):
            data_type_index = np.where(data_type == n)  # index value for n centroid
            if len(data_type_index[0]) != 0:  # next if n centroid dont have value
                self.centroids[n] = np.average(self.data[data_type_index], axis=0)

    def plot(self):
        plot_k_means(self.data, self.centroids, color=self.data_type, fig_type="K_means")

    def save(self):
        self.save_path = f'models/K_means_{self.classes_num}_{self.dimension}.csv'
        np.savetxt(self.save_path, self.save_data, delimiter=',')
