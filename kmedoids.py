import math
import pandas as pd
import numpy as np
np.set_printoptions(precision=3)


# read in data set
def read_data(csv_name, seper=','):
    # read in from pandas, then convert to matrix
    data = pd.read_csv(csv_name, sep=seper).values
    return data


# compute distance function
def calc_dist(x1, x2):
    Z = x1-x2
    dist = np.sqrt(sum([z*z for z in Z]))
    # print('distance: {}\n\t for: {}'.format(dist, Z))
    return dist


def build(data, k):
    # initialize variables
    TD = math.inf  # total deviation
    M = [(math.inf, math.inf) for kk in range(k)]  # set of medoids
    M_index = [math.inf for kk in range(k)]  # hold indicies for values
    # find first medoid
    for i, xi in enumerate(data):
        tdj = 0
        # itterate through all data points that are not equal to xi
        for o in [ii for ii in range(len(data)) if ii != i]:
            tdj = tdj + calc_dist(xi, data[o])
            if tdj < TD:
                TD, M[0] = tdj, xi
                M_index[0] = i
                # loc_in_M = i
                print(M)
                print('assigned new TD: {} for xi: {}'.format(TD, i))
    # del data[loc_in_M]
    for i in range(1, k):
        cTD = math.inf  # change in TD
        for j in [jj for jj in range(len(data)) if jj not in M_index]:
            xj = data[j]
            ctd = 0  # test td
            for o in [oo for oo in range(len(data)) if oo not in M_index or
                      M != j]:
                xo = data[o]
                # print('taking min from: ', M[0:i])
                min_mo = min([calc_dist(xo, mm) for mm in M[0:i]])
                delta = calc_dist(xj, xo) - min_mo
                if delta < 0:
                    ctd = ctd + delta
            if ctd < cTD:
                cTD = ctd
                M[i] = xj
                M_index[i] = j
                print('added new m: {} at {} with cTD {}'.format(xj, j, cTD))
        TD = cTD + TD

    # print final medoids
    print('\n\nFinal TD: {} medoids: {}'.format(TD, M))
    print('Located at: {}'.format(M_index))


def swap(data, TD, M):
    # TODO: Put code for swap method here
    pass


if __name__ == '__main__':
    print('Running as main class')
    k = 2  # number of clusters
    data = read_data('simple11.csv', seper=' ')
    build(data, k)
