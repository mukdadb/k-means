import matplotlib.pyplot as plt

from kmeans import kmeans

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
with open('sample.txt', 'r') as f:
    pairs = [line.split(' ') for line in f.readlines()]
    sample = [[float(pair[0]), float(pair[1])] for pair in pairs]

result = kmeans(sample, 2)

for cluster, color in zip(result, COLORS):
    plt.plot([point[0] for point in cluster], [point[1] for point in cluster],
             '.' + color)
plt.show()
