import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances as euclid
import normalized_cut as nc


def calc_cluster_means(data, clusters):
    means = []
    for c in clusters:
        if len(c) != 0:
            v = np.sum([np.array(data[i], dtype=float) for i in c],
                       axis=0)/len(c)
            means.append(v)
        else:
            means.append(0)
    return means


def random_initial_means(data, k):
    # find initial means, random points
    init_mean = np.random.choice(range(len(data)), size=k, replace=False)
    means = [data[init_mean[i]] for i in range(k)]
    return means


def cluster(data, means, k, error):
    # keep track of old means
    track_means = [means]

    exp_error = math.inf  # intital experimental error to infinity
    # itterate while experimental error is greater then error
    while exp_error > error:
        # make empty cluster array
        clusters = [[] for x in range(k)]

        # find closest mean for all points
        for i, x in enumerate(data):
            dist = []

            # calculate distance to each mean
            for d in means:
                dist.append(np.linalg.norm(x-d))

            # append to the cluster with the closest mean
            clusters[np.argmin(dist)].append(i)

        means = calc_cluster_means(data, clusters)

        # calculate error
        exp_error = 0
        for m in range(len(means)):
            exp_error += np.linalg.norm(means[m] - track_means[-1][m])**2

        # append new means to tracking
        track_means.append(means)

        # break loop if too many itterations
        if len(track_means) > 5:
            break
    return means, clusters, track_means


if __name__ == '__main__':
    print('running as main')
    data = pd.read_csv('atom.csv', sep=' ', header=None).values
    k = 2
    # means = random_initial_means(data, k)
    # means = calc_cluster_means(data, [[0], [8]])
    means = [[0.1464,  -2.5902, -49.4474], [0.104272, -0.1783, 1.7890]]
    error = .001**2*k  # each mean can only change by .001
    means, clusters, tm = cluster(data, means, 2, error)
    print(means)
    print(clusters)
    print(len(tm))
    distm = euclid(data)
    cut = nc.calculate(clusters, distm)
    print('nc: ', cut)
