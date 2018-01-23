import numpy as np
from scipy.spatial.distance import euclidean


def _init_centroids(points, k):
    c = [None] * k  # Centroids coords
    d = [None] * len(points)  # Points' distances from nearest centroid
    c[0] = points[np.random.choice(len(points))]
    for i in range(1, k):
        for j, point in enumerate(points):
            d[j] = min([euclidean(point, centroid) for centroid in c[:i]])
        p = [distance ** 2 for distance in d]
        psum = sum(p)
        p = [x / psum for x in p]
        c[i] = points[np.random.choice(len(points), p=p)]
    return c


def _update_centroids(points, centroids, assignment):
    new_c = [None] * len(centroids)
    for i, centroid in enumerate(centroids):
        assigned_points = [points[j]
                           for j, point in enumerate(points)
                           if assignment[j] == i]
        new_c[i] = list(np.array(assigned_points).mean(axis=0))
    return new_c


def kmeans(points, k):
    """K-means clustering for given k. Returns array of groups."""
    # Centroids initialization
    c = _init_centroids(points, k)
    # Main part
    while True:
        # Assignment
        assignment = [None] * len(points)  # index: point i, value: centroid i
        for i, point in enumerate(points):
            assignment[i] = c.index(min(c,
                                    key=lambda cntr: euclidean(point, cntr)))
        # Centroid update
        new_c = _update_centroids(points, c, assignment)
        # Stop condition
        if np.array_equal(np.array(c), np.array(new_c)):
            break
        c = new_c
    return np.array([[points[j]
                    for j, point in enumerate(points) if assignment[j] == i]
                    for i in range(len(c))])
