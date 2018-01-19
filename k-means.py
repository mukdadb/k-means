import numpy as np
from scipy.spatial.distance import euclidean

def kmeans(points, k):
    '''K-means clustering for given k. Returns array of groups.'''
    # Centroids initialization
    c = [None] * k            # Centroids coords
    d = [None] * len(points)  # Points' distances from nearest centroid
    c[0] = points[np.random.choice(len(points))]
    for i in range(1,k):
        for j, point in enumerate(points):
            d[j] = min([euclidean(point, centroid) for centroid in c[:i]])
        p = [distance ** 2 for distance in d]
        psum = sum(p)
        p /= psum
        c[i] = points[np.random.choice(len(points), p=p)]
    # Main part
    assigment = [None] * len(points) # index: point index, val: centroid index
    while True:
        # Assigment
        assigment = [None] * len(points)
        for i, point in enumerate(points):
            assigment[i] = c.index(min(c,
                                       key=lambda cntr: euclidean(point,cntr)))
        # Centroid update
        new_c = [None for cntr in c]
        for i, centroid in enumerate(c):
            assigned_points = [points[j] for j, point in enumerate(points)
                                                        if assigment[j] == i]
            new_c[i] = list(np.array(assigned_points).mean(axis=0))
        # Stop condition
        if np.array_equal(np.array(c), np.array(new_c)):
            break
        c = new_c
    print(c)
    return [[points[j] for j, point in enumerate(points) if assigment[j] == i]
                                                        for i in range(len(c))]



print(kmeans([[1,1],[0,0],[5,5],[10,10], [9,9]], 2))