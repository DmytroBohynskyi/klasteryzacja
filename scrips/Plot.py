import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_3d(data, centroids, color):
    fig = plt.figure()
    for i in range(data.shape[1] - 2):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        ax.scatter(data[:, i], data[:, i + 1], data[:, i + 2],
                   c=color, s=40, cmap='Accent_r', marker='.')
        plt.scatter(centroids[:, i], centroids[:, i + 1], centroids[:, i + 2],
                    c='black', marker='o')


def plot_k_means(data, centroids, color='gray', end=True, fig_type="None"):
    if data.shape[1] is 2:
        plt.scatter(data[:, 0], data[:, 1], c=color, s=40, cmap='Accent_r', label='data points')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", color='black', label='centroids')
    elif data.shape[1] > 2:
        plot_3d(data, centroids, color)

    if end:
        plt.savefig(f"images/{fig_type}_{data.shape[1]}_{centroids.shape[1]}")
        plt.show()
