import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
# take in clusters, distance matrix and determine score

def calculate(clust, distm):
    NC = 0

    # itterate through k times
    for c in clust:
        # calculate win
        win = 0
        for cc in c:
            for cc2 in range(cc, len(c)):
                win += distm[cc][cc2]

        # make array not in c
        others = [dd for dd in range(len(distm)) if dd not in c]
        wout = 0
        for cc in c:
            for o in others:
                wout += distm[cc][o]

        # add to sum for this itteration of k
        NC += 1/(win/wout + 1)

    print('NC = ', NC)
    return NC


def read_in_files(cluster_csv, distm_csv):
    pass


def simple_test():
    data = pd.read_csv('simple01.csv', sep=' ', header=None)
    print(data)
    print(data.columns.values)
    data = data.values

    # print([str(m) + '\n' for m in distm])
    # calc dist matrix
    distm = np.ndarray(shape=(len(data), len(data)), dtype=float)
    for i in range(len(data)-1):
        for j in range(i, len(data)):
            distm[i][j] = np.linalg.norm(data[i]-data[j])
            # print('dist {} {} = {}'.format(i, j, distm[i][j]))
            distm[j][i] = distm[i][j]

    print(distm)
    print(len(distm))
    c = [[0,1,2,3,4], [5,6,7,8,9]]
    calculate(c, distm)


if __name__ == '__main__':
    simple_test()
