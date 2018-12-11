from datetime import datetime as dt
import math
import pandas as pd
import numpy as np
import normalized_cut as nc
from sklearn.metrics.pairwise import euclidean_distances as euclid


# set global options
# np.set_printoptions(precision=3)
start_time = dt.now()
use_dist_matrix = False


# read in data set
def read_data(csv_name, seper=',', headers='infer'):
    print('\nReading in csv: {}'.format(csv_name))
    csv_start_time = dt.now()

    # read in from pandas, then convert to matrix
    data = pd.read_csv(csv_name, sep=seper, header=headers).values
    print('Done reading {}'.format(str(dt.now()-csv_start_time)))
    return data


# compute distance function
def calc_dist(x1, x2):
    Z = x1-x2
    dist = np.sqrt(sum([z*z for z in Z]))
    # print('distance: {}\n\t for: {}'.format(dist, Z))
    return dist


def build(data, k, distm=0):
    print('\nRunning build method.....')

    # initialize variables
    TD = math.inf  # total deviation
    M = [(math.inf, math.inf) for kk in range(k)]  # set of medoids
    M_index = [math.inf for kk in range(k)]  # hold indicies for values

    # find first medoid
    print('\t Initial medoid.....')
    for i, xi in enumerate(data):
        if i % 1000 == 0:
            print('\t\tbuild point: {}'.format(i))
        tdj = 0
        # itterate through all data points that are not equal to xi
        for o in [ii for ii in range(len(data)) if ii != i]:
            tdj = tdj + distm[i][o]
            if tdj < TD:
                TD, M[0] = tdj, xi
                M_index[0] = i
                # loc_in_M = i
                # print(M)
                # print('assigned new TD: {} for xi: {}'.format(TD, i))

    # del data[loc_in_M]
    for i in range(1, k):
        print('\tMedoid {}.....'.format(i))
        cTD = math.inf  # change in TD
        for j in [jj for jj in range(len(data)) if jj not in M_index]:
            if j % 1000 == 0:
                print('\t\tPoint: {} of {}'.format(j, len(data)))
            xj = data[j]
            ctd = 0  # test td
            for o in [oo for oo in range(len(data)) if oo not in M_index or
                      M != j]:
                # xo = data[o]
                # print('taking min from: ', M[0:i])
                min_mo = min([distm[o][mm] for mm in M_index[0:i]])
                delta = distm[o][j] - min_mo
                if delta < 0:
                    ctd = ctd + delta
            if ctd < cTD:
                cTD = ctd
                M[i] = xj
                M_index[i] = j
                # print('added new m: {} at {} with cTD {}'.format(xj, j, cTD))
        TD = cTD + TD

    # print final medoids
    # print('\n\nFinal TD: {} medoids: {}'.format(TD, M))
    # print('Located at: {}'.format(M_index))
    print('\tRuntime for build: {}'.format(dt.now()-start_time))

    return TD, (M, M_index)


# create distance table
def build_dist_table(data, M, M_index, distm):
    # print('Creating a distance table....')
    dist_table = []
    dist_index = []
    for j, x in enumerate(data):
        temp_m = []
        for i, m in enumerate(M):
            temp_m.append(distm[j][i])
        first = min(temp_m)
        first_index = temp_m.index(first)
        # set the first to inf, to get the second min
        temp_m[first_index] = math.inf
        second = min(temp_m)
        second_index = np.argmin(temp_m)  #
        dist_table.append([first, second])
        dist_index.append([first_index, second_index])

    return dist_table, dist_index


def compare_medoids(index_list):
    # print('old {} new {}'.format(M_old, M_new))
    for m in index_list[-1]:  # compare the new list to:
        if m in index_list[-2] or index_list[-3]:
            pass
        else:
            # not all values are the same
            print('{} is new'.format(m))
            return False
    # made it through loop, medoids same as last run
    print('all medoids are the same')
    return True


# PAM+
def swap(data, TD, Med, Dist, distm):
    print('\nRunning swap method....')

    # create needed variables
    M, M_index = Med
    dist, dist_index = Dist
    old_index_M = [[], [], M_index]

    # get k, assuming will need k itterations to find a good cluster
    kp = len(Med[0]) + 1

    # the algorithm just says repeat.....
    # while not compare_medoids(old_index_M):
    for itter in range(kp):
        print('\trun: ', M_index)
        cTD = 0
        m_temp, x_temp = 0, 0  # store temp indicies
        for j in [jj for jj in range(len(data)) if jj not in M_index]:
            if j % 1000 == 0:
                print('\t\tLooking at data point: {}'.format(j))
            dj = dist[j][0]
            ctd = [-dj for jj in range(len(M))]  # add new dist for each medoid
            for o in [oo for oo in range(len(data)) if oo != j]:
                doj = distm[o][j]
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
                # print('\t\t\t', j, ' breaking')
                # print('\t\tcTD is greater then 0, breaking the loop')
                # break
                continue
            # else:
                # print('{} is not > 0'.format(j))
        # swap roles of m_temp and x_temp:w
        M[m_temp] = data[x_temp]
        M_index[m_temp] = x_temp
        dist, dist_index = build_dist_table(data, M, M_index, distm)
        print('\t\tFinished swap itteration, total runtime {}'.format(
              dt.now()-start_time))
        TD = TD + cTD
        # add index set to list
        old_index_M.append(M_index.copy())
    return TD, (M, M_index), (dist, dist_index)


def cluster_from_dist(dist_index, k):
    clusters = [[] for x in range(k)]
    for i, d in enumerate(dist_index):
        clusters[d[0]].append(i)
    return clusters


def print_info(data, M, C, output=''):
    output += str(M[0])
    for i in range(1, len(M)):
        output += ', ' + str(M[i])
    output += '\n'

    for i, c in enumerate(C):
        temp_c = [str(data[ci]) for ci in c]
        output += 'C' + str(i) + ': ' + ', '.join(temp_c) + '\n'

    return output


def test_run(csv_name, k, data=0):
    if isinstance(data, int):
        data = read_data(csv_name, seper=' ', headers=None)
    # build method
    TD, Med = build(data, k)
    print('initial medoids: {} \n\t {}'.format(Med[1], Med[0]))
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    # swap method
    TD, Med, Dist = swap(data, TD, Med, Dist)
    # get clusters
    clus = cluster_from_dist(dist_i, k)
    # record information
    info = csv_name + '\n'
    out = ''.join(['***' for x in range(20)]) + '\n'
    out += print_info(data, Med[0], clus, info)

    # add cluster numbers
    out += '\n'
    for c in clus:
        temp_c = [str(cc) for cc in c]
        out += ','.join(temp_c) + '\n'

    out += '\n\n'
    print(out)
    with open("test-output.txt", "a") as myfile:
        myfile.write(out)


def cluster(data, k):
    TD, Med = build(data, k)
    print('initial medoids: {} \n\t {}'.format(Med[1], Med[0]))
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    # swap method
    TD, Med, Dist = swap(data, TD, Med, Dist)
    # get clusters
    clus = cluster_from_dist(dist_i, k)
    return Med[1], clus


def output_clusters(clusters, file_name='none'):
    # print clusters to csv
    print_clusters = ''
    for c in clusters:
        print_clusters += ','.join([str(cc) for cc in c]) + '\n'

    """
    # write to file
    output_name = 'medoid-test-run-' + str(start_time) + '.csv'
    with open(output_name, 'w') as f:
        f.write(print_clusters)
    """
    return print_clusters


def track(data, k, csv_name='undeclared'):
    # create initial file
    output_name = 'runs/kmedoid-' + str(start_time)
    with open(output_name, 'a') as f:
        out = 'Testing kmedoids.py on ' + csv_name + ' where k =  ' + str(k)
        out += ' with' + str(len(data)) + ' instances\n'
        f.write(out)

    # calculate distance matrix
    distm = euclid(data)

    # get intial medoids
    TD, Med = build(data, k, distm)
    with open(output_name, 'a') as f:
        out = 'initial TD = ' + str(TD) + ' Medoids: ' + str(Med[1]) + '\n'
        f.write(out)

    # swap method
    Dist = build_dist_table(data, Med[0], Med[1], distm)  # table, index
    TD, Med, Dist = swap(data, TD, Med, Dist, distm)

    # get clusters
    clus = cluster_from_dist(Dist[1], k)

    out = 'Final TD: ' + str(TD) + ' Medoids: ' + str(Med[1]) + '\n'
    out += 'Clusters are: \n'
    out += output_clusters(clus)

    with open(output_name, 'a') as f:
        f.write(out)

    # calculate normalized cut
    cut = nc.calculate(clus, distm)
    out = 'Normalized cut: ' + str(cut) + '\n'
    out += '\nTotal run time: ' + str(dt.now() - start_time)
    with open(output_name, 'a') as f:
        f.write(out)

    return Med[1], clus


if __name__ == '__main__':
    print('Running as main class')
    # test_run('simple01.csv', 2)
    # test_run('atom.csv', 2)
    # test_run('simple11.csv', 2)
    # test_run('iris.csv', 3)
    """
    data = pd.read_csv('atom.csv', sep=' ', header=None).values
    m, c = cluster(data, 2)
    NC = nc.calculate(c, make_dist_matrix(data))
    print(NC)
    """

    k = 2
    file_name = 'full-monthly-avgs.csv'
    file_name = 'monthly_avg_zscore.csv'
    data = pd.read_csv(file_name).iloc[0:, 3:8]
    print('headers: {}'.format(data.columns.values))
    data = data.values

    # distm = make_dist_matrix_and_file(data, 'avg-data3-8-dist-matrix.csv')

    # use faster method to calculate distance matrix....
    distm = euclid(data)

    data = data[0:2000]

    track(data, k, file_name)
    """
    # read in distance matrix from file
    # distm = read_data('avg-data3-8-dist-matrix.csv', headers=None)
    TD, Med = build(data, k, distm)
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    TD, Med, Dist = swap(data, TD, Med, Dist, distm)
    clus = cluster_from_dist(dist_i, k)
    print('c1: {} c2: {}'.format(len(clus[0]), len(clus[1])))
    output_clusters(clus)

    # calc nc on clusters
    cut = nc.calculate(clus, distm)
    print(cut)
    """
    # data = read_data('simple01.csv', ' ')
    # test_run('full-monthly-avgs.csv', 2, data)
    # dm = make_dist_matrix_and_file(data, 'avg-data2-8-dist-matrix.csv')
    # print('len dm: {}'.format(len(dm)))
    # print(dm[0:4][0:5])

    """
    TD, Med = build(data, k)
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    TD, Med, Dist = swap(data, TD, Med, Dist)
    clus = cluster_from_dist(dist_i, k)
    print('c1: {} c2: {}'.format(len(clus[0]), len(clus[1])))

    output = file_name + ' with ' + str(k) + ' clusters and a runtime of '
    output += str(dt.now() - start_time) + '\n'
    out_file_name = file_name + '-' + str(start_time)
    with open(out_file_name, "a") as myfile:
        myfile.write(print_info(data, Med[0], clus, output))
    """

    """
    k = 2  # number of clusters
    data = read_data('simple01.csv', seper=' ')
    TD, Med = build(data, k)
    print('med: ', Med)
    # Med[0][0],Med[1][0] = data[6],6
    # Med[0][1],Med[1][1] = data[7],7
    # print(Med)
    dist_table, dist_i = build_dist_table(data, Med[0], Med[1])
    Dist = (dist_table, dist_i)
    # print(dist_table)
    # print(dist_i)
    print('build TD: {0:.3f}'.format(TD))
    TD, Med, Dist = swap(data, TD, Med, Dist)
    print('\n\nfinal TD: {0:.3f}'.format(TD))
    # print(Dist[1])
    print('\nmed ', Med)
    print('\n\n\n')
    clus = cluster_from_dist(dist_i, k)
    print('len: ', len(clus))
    print(clus)
    out = print_info(data, Med[0], clus, 'simple01.csv\n')
    print(out)
    """
