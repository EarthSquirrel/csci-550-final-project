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

    return TD, (M, M_index)


# create distance table
def build_dist_table(data, M, M_index):
    dist_table = []
    dist_index = []
    for x in data:
        temp_m = []
        for i, m in enumerate(M):
            temp_m.append(calc_dist(x, m))
        first = min(temp_m)
        first_index = temp_m.index(first)
        # set the first to inf, to get the second min
        temp_m[first_index] = math.inf
        second = min(temp_m)
        second_index = np.argmin(temp_m)  #
        dist_table.append([first, second])
        dist_index.append([first_index, second_index])

    return dist_table, dist_index


# PAM+
def swap(data, TD, Med, Dist):
    M, M_index = Med
    dist, dist_index = Dist
    # the algorithm just says repeat.....
    for x in range(1):
        cTD = 0
        m_temp, x_temp = 0, 0  # store temp indicies
        for j in [jj for jj in range(len(data)) if jj not in M_index]:
            dj = dist[j][0]
            ctd = [-dj for jj in range(len(M))]  # add new dist for each medoid
            for o in [oo for oo in range(len(data)) if oo != j]:
                doj = calc_dist(data[o], data[j])
                n = dist_index[o][0]
                dn, ds = dist[o]  # add first and second nearest dist
                ctd[n] = ctd[n] + min([doj, ds]) - dn
                if doj < dn:
                    for i in [ii for ii in range(len(M)) if ii != n]:
                        ctd[i] = ctd[i] + doj - dn
            i = np.argmin(ctd)
            if ctd[i] < cTD:
                cTD = ctd[i]
                m_temp = i
                x_temp = j
            if cTD >= 0:
                break
        # swap roles of m_temp and x_temp:w
        M[m_temp] = data[x_temp]
        M_index[m_temp] = x_temp
        dist, dist_index = build_dist_table(data, M, M_index)

        TD = TD + cTD
    return TD, (M, M_index), (dist, dist_index)


def cluster_from_dist(dist_index, k):
    clusters = [[] for x in range(k)]
    for i, d in enumerate(dist_index):
        clusters[d].append(i)
    return clusters


if __name__ == '__main__':
    print('Running as main class')
    k = 2  # number of clusters
    data = read_data('simple01.csv', seper=' ')
    TD, Med = build(data, k)
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    print(dist_table)
    print(dist_i)
    print('build TD: {0:.3f}'.format(TD))
    TD, Med, Dist = swap(data, TD, Med, Dist)
    print('\n\nfinal TD: {0:.3f}'.format(TD))
    print(Dist[1])
