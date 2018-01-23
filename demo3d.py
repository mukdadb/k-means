import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from kmeans import kmeans

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

sample = [list(arr) for arr in 100 * np.random.random_sample((100, 
3))]

result = kmeans(sample, 7)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, cluster in zip(COLORS, tuple(result)):
    xs = [point[0] for point in cluster]
    ys = [point[1] for point in cluster]
    zs = [point[2] for point in cluster]
    ax.scatter(xs, ys, zs, c=c)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
