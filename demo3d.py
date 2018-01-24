import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from kmeans import kmeans

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

mu, kappa = 2.5, 2.0
sample = [list(arr) for arr in 100 * np.random.vonmises(mu, kappa, (30, 3))]

result = kmeans(sample, 6)

fig = plt.figure()
bx = fig.add_subplot(111, projection='3d')
plt.title('Raw data')

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
plt.title('Clustered data')

for c, cluster in zip(COLORS, tuple(result)):
    xs = [point[0] for point in cluster]
    ys = [point[1] for point in cluster]
    zs = [point[2] for point in cluster]
    ax.scatter(xs, ys, zs, c=c)
    bx.scatter(xs, ys, zs, c='k')

while True:
    for angle in range(0, 360):
        ax.view_init(30, angle)
        bx.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
