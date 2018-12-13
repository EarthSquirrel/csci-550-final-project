import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
# take in clusters, distance matrix and determine score

def calculate(clust, distm):
    NC = 0

    # itterate through k times
    # c is a cluster
    for i, c in enumerate(clust):
        # calculate win
        win = 0
        for cci, cc in enumerate(c):
            # cc is index of value in cluster
            # print('cc {} c {}'.format(type(cc), type(c)))
            for cc2 in range(cci, len(c)):
                win += distm[cc][cc2]
        win = win/2

        # make array not in c
        others = [dd for dd in range(len(distm)) if dd not in c]
        wout = 0
        for cc in c:
            for o in others:
                wout += distm[cc][o]
        wout = wout/2

        # add to sum for this itteration of k
        NC += 1/(win/wout + 1)
        # print('{}:: win: {} wout: {}'.format(i, win, wout))

    print('NC = ', NC)
    return NC


def read_in_files(cluster_csv, distm_csv):
    clus = pd.read_csv(cluster_csv, header=None)
    distm = pd.read_csv(distm_csv, header=None)
    return clus, distm


def simple_test():
    data = pd.read_csv('simple01.csv', sep=' ', header=None)
    data = data.values

    # print([str(m) + '\n' for m in distm])
    # calc dist matrix
    distm = np.ndarray(shape=(len(data), len(data)), dtype=float)
    for i in range(len(data)-1):
        for j in range(i, len(data)):
            distm[i][j] = np.linalg.norm(data[i]-data[j])
            # print('dist {} {} = {}'.format(i, j, distm[i][j]))
            distm[j][i] = distm[i][j]

    # print(distm)
    # print(len(distm))
    c = [[0,1,2,3,4], [5,6,7,8,9]]
    print(c)
    calculate(c, distm)
    c = [[5, 6, 8], [0, 1, 2, 3, 4, 7, 9]]
    print(c)
    calculate(c, distm)


def find_column_variance():
    data = pd.read_csv('full-monthly-avgs.csv').iloc[0:, 3:8]
    print(data.columns.values)
    data = data.values
    # data = pd.read_csv('monthly_avg_zscore.csv').iloc[0:, 3:8].values
    print('len data: ', len(data))
    t = np.transpose(data)
    print('transposed len: ', len(t))
    for r in t:
        print('std: ', np.std(r))


if __name__ == '__main__':
    # simple_test()
    find_column_variance()
