import pandas as pd
import numpy as np
from datetime import datetime as dt
import math
from sklearn.metrics.pairwise import euclidean_distances as euclid
import normalized_cut as nc
import random
start_time_in = dt.now()
start_time = dt.now()


def calc_cluster_means(data, clusters):
    means = []
    for c in clusters:
        if len(c) != 0:
            v = np.sum([np.array(data[i], dtype=float) for i in c],
                       axis=0)/len(c)
            means.append(v)
        else:
            means.append(np.array([0 for z in range(len(data[0]))],
                         dtype=float))
    return means


def random_initial_means(data, k):
    # find initial means, random points
    init_mean = np.random.choice(range(len(data)), size=k, replace=False)
    means = [data[init_mean[i]] for i in range(k)]
    return means


def scramble_start_means(data, k):
    scrambled = data.copy()
    by_feature = np.transpose(scrambled)
    for f in range(len(by_feature)):
        by_feature[f].sort()

    # divinde into chucks
    chunk_len = round(len(by_feature[0])/k)
    midpoints = [[] for x in range(len(by_feature))]
    for i in range(0, len(data)-chunk_len, chunk_len):
        mid_index = int(i + round(.5*chunk_len, 0))
        # append midpoint for each feature
        for f in range(len(by_feature)):
            midpoints[f].append(by_feature[f][mid_index])

    # randomly select means
    means = []
    for j in range(k):
        m = []
        for f in range(len(by_feature)):
            m.append(random.choice(midpoints[f]))
        means.append(m)
    return means


# create k matricies with sse for each point
def distance_matricies(data, means):
    distms = []
    for m in means:
        # convert to array for subtraction
        m = np.array(m, dtype=float)
        sub = []
        for x in data:
            x = np.array(m, dtype=float)
            sub.append(x-m)
            print('appended: ', sub[-1])

    distms.append(sub)


def calc_sse(data, means, clusters):
    sse = 0
    for j, m in enumerate(means):
        m = np.array(m, dtype=float)
        d = np.array([data[i] for i in clusters[j]], dtype=float)
        sse += np.linalg.norm(d-m)**2
        # print('new sse:: ', sse)

    return sse


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
                dist.append(np.linalg.norm(x-d)**2)

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
        if len(track_means) > 200:
            print('breaking, more than 200 itterations')
            break

    # print('itterations: ', len(track_means))
    return means, clusters, track_means


def track_data(data, means, k, error, distm, file_name, notes='', cnote='\n'):
    with open(file_name, 'a') as f:
        out = ''.join(['***' for x in range(15)]) + '\n'
        out += 'Running kmeans where k = ' + str(k) + ' with ' + str(len(data))
        out += ' instances\n' + notes + '\n'
        out += 'Initial means: ' + str(means) + '\n'
        f.write(out)

    m, c, tm = cluster(data, means, k, error)
    cut = ''
    try:
        cut = nc.calculate(c, distm)
    except ZeroDivisionError:
        print('NC divinding by 0')
        cut = k
    sse = ''
    try:
        sse = calc_sse(data, m, c)
    except ValueError:
        print('ValueError for sse')
        sse = 'ValueError'

    out = 'Total itteratioins: ' + str(len(tm)) + '\n'
    out += 'Total runtime: ' + str(dt.now() - start_time) + ' in '
    out += str(len(tm)) + ' iterations.\n'
    out += 'NC: ' + str(cut) + '\n'
    out += 'SSE: ' + str(sse) + '\n'
    out += 'Tracked means: ' + str(tm) + '\n\n'
    out += 'Final means: ' + str(m) + '\n\n\n'

    print_clusters = '' + str(len(data))
    for i in c:
        print_clusters += ','.join([str(cc) for cc in i]) + '\n'
    out += print_clusters + '\n\n'

    with open(file_name, 'a') as f:
        f.write(out)
    """
    # make kmeans k tracking file
    # with open('runs/kmeans/track-k-nc-cut.txt', 'a') as f:
    with open('runs/kmeans/track-k-scram-nc-cut.txt', 'a') as f:
        out = str(k) + ',' + str(cut) + '\n'
        f.write(out)
    """
    # make kmeans k tracking file sse
    # with open('runs/kmeans/track-k-sse-cut.txt', 'a') as f:
    with open('runs/kmeans/track-k-scram2-sse-cut.txt', 'a') as f:
        out = str(k) + ',' + str(sse) + '\n'
        f.write(out)
    """
    # write clusters specifically to file
    # with open('runs/kmeans/clusters-means-cut.txt', 'a') as f:
    with open('runs/kmeans/clusters-scram2-means-cut.txt', 'a') as f:
        out = '\n' + cnote + print_clusters
        f.write(out)
    """

    print('k: ', k, ' iter: ', len(tm),  ' sse: ', sse)


if __name__ == '__main__':
    print('running as main at ', start_time_in)

    data = pd.read_csv('monthly_avg_zscore.csv').iloc[0:, 3:8].values
    data_dm = euclid(data)

    print('set up complete: starting to cluster\n')
    file_name = 'runs/kmeans/test-scram' + str(len(data)) + '-'
    # file_name = 'runs/kmeans/test' + str(len(data)) + '-'
    file_name += str(start_time_in)
    # for k in range(2, 10):
    #    for i in [0,0,0]:
    #        print(scramble_start_means(data, k))

    k = 2
    notes = ' scram-means '
    for k in range(2, 62):
        start_time = dt.now()
        error = .05**2*k  # each mean can only change by .001
        # means = random_initial_means(data, k)
        means = scramble_start_means(data, k)
        track_data(data, means, k, error, data_dm, file_name, notes)

    print('final run time: ', (dt.now() - start_time_in))
