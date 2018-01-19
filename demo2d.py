import matplotlib.pyplot as plt
import numpy as np

from kmeans import kmeans

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

sample = [list(arr) for arr in 100 * np.random.random_sample((1000, 2))]

result = kmeans(sample, 6)

for cluster, color in zip(result, COLORS):
    plt.plot([point[0] for point in cluster], [point[1] for point in cluster],
             '.' + color)
plt.show()
