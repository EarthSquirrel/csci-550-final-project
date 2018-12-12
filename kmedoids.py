from datetime import datetime as dt
import math
import pandas as pd
import numpy as np
import normalized_cut as nc
from sklearn.metrics.pairwise import euclidean_distances as euclid


# set global options
# np.set_printoptions(precision=3)
start_time = dt.now()


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


def build(data, k, distm):
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
        tdi = 0
        # itterate through all data points that are not equal to xi
        for o in [ii for ii in range(len(data)) if ii != i]:
            tdi = tdi + distm[i][o]
        if tdi < TD:
            TD, M[0] = tdi, xi
            M_index[0] = i
    print('initital TD ', TD)
    with open('runs/FINAL-INITIAL-MEDS.txt', 'a') as f:
        out = str(M_index[0]) + '\n'
        f.write(out)

    # start at later medoid, k = 14
    # M_index = [2304, 6435, 993, 5790, 2379, 11328, 181, 4461, 376, 2090,
    #            10137, 8858, 1826, 2953]
    # M = [data[i] for i in M_index]
    # for mm in range(1, k):
    #    M_index.append((math.inf, math.inf))
    #    M.append(math.inf)
    # print('len Ms: {} {}'.format(len(M_index), len(M)))
    # TD = -21960.494001124956

    for i in range(1, k):
        print('\tMedoid: ', i, '.....')
        cTD = math.inf  # change in TD
        # go through each point not in m
        for j in [jj for jj in range(len(data)) if jj not in M_index]:
            if j % 1000 == 0:
                print('\t\tPoint: {} of {}'.format(j, len(data)))
            xj = data[j]
            ctd = 0  # test td
            # compare to all other points
            for o in [oo for oo in range(len(data)) if oo not in M_index or
                      M != j]:
                min_mo = min([distm[o][mm] for mm in M_index[0:i]])
                delta = distm[o][j] - min_mo
                if delta < 0:
                    ctd = ctd + delta
            if ctd < cTD:
                cTD, M[i], M_index[i] = ctd, xj, j
        TD = cTD + TD

        # print to file so can stop and not lose as much
        time_too_long = str(dt.now() -start_time)
        with open('runs/FINAL-INITIAL-MEDS.txt', 'a') as f:
            out = str(i) + ' ' + time_too_long
            out += '\n' + str(TD) + '\n' + str(M_index[0:i+1]) + '\n\n'
            # print(out)
            f.write(out)

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

    inter_med = []  # hold the middle medoids
    # get k, assuming will need k itterations to find a good cluster
    kp = len(Med[0]) + 3

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
                break
                # continue
            # else:
                # print('{} is not > 0'.format(j))
        # swap roles of m_temp and x_temp:w
        M[m_temp] = data[x_temp]
        M_index[m_temp] = x_temp
        dist, dist_index = build_dist_table(data, M, M_index, distm)
        TD = TD + cTD
        # add index set to list
        old_index_M.append(M_index.copy())
        inter_med.append(M_index.copy())  # track medoid chain

    print('\tFinished swap itteration, total runtime {}'.format(
          dt.now()-start_time))

    return TD, (M, M_index), (dist, dist_index), inter_med


def cluster_from_dist(dist_index, k):
    clusters = [[] for x in range(k)]
    for i, d in enumerate(dist_index):
        clusters[d[0]].append(i)
    return clusters


# used to compare test data with pyclustering
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


def track(data, k, csv_name='undeclared', notes=''):
    # create initial file
    output_name = 'runs/kmedoid-' + str(start_time)
    with open(output_name, 'a') as f:
        out = 'Testing kmedoids.py on ' + csv_name + ' where k =  ' + str(k)
        out += ' with' + str(len(data)) + ' instances\n'
        out += notes + '\n'
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
    TD, Med, Dist, inter = swap(data, TD, Med, Dist, distm)

    # get clusters
    clus = cluster_from_dist(Dist[1], k)

    out = 'Progress medoids: ' + str(inter) + '\n'
    out += 'Final TD: ' + str(TD) + ' Medoids: ' + str(Med[1]) + '\n'
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

    # write special k to nc comparison file
    with open('runs/k-nc-comparison.txt', 'a') as f:
        out = str(k) + ',' + str(cut) + '\n'
        f.write(out)

    return Med[1], clus


def calc_td(data, M_index, dist_i, distm):
    TD = 0
    cluster = cluster_from_dist(dist_i, len(M_index))
    for k in range(len(M_index)):
        for c in cluster[k]:
            TD += distm[c][k]
    return TD


def track_swap(data, med_list, file_name):
    M_full = [data[m] for m in med_list]  # data points
    start_TD = []
    final_TD = []
    for i in range(2, len(med_list)):
        M = M_full[0:i]
        M_index = med_list[0:i]
        Dist = build_dist_table(data, M, M_index, distm)
        TD = calc_td(data, M_index, Dist[1], distm)
        print('', i, ' td: ', TD)
        start_TD.append(TD)
        Med = (M, M_index)
        TDf, Med, Dist, tm = swap(data, TD, Med, Dist, distm)
        final_TD.append(TDf)
        print_clusters = output_clusters(cluster_from_dist(Dist[1], i))
        with open('full-test.txt', 'a') as f:
            out = '***'.join(['***' for ccc in range(15)]) + '\n'
            out += 'Start TD: ' + str(TD) + ' end TD: ' + str(TDf) + '\n'
            out += print_clusters + '\n'
            f.write(out)
        with open('full-clusters.txt', 'a') as f:
            out = print_clusters + '\n\n'
            f.write(out)
    with open('full-test.txt', 'a') as f:
        f.write(str(final_TD))


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
    file_name = 'monthly_avg_zscore.csv'
    data = pd.read_csv(file_name).iloc[0:, 3:8]
    print('headers: {}'.format(data.columns.values))
    data = data.values

    # distm = make_dist_matrix_and_file(data, 'avg-data3-8-dist-matrix.csv')

    # use faster method to calculate distance matrix....
    distm = euclid(data)

    # for k in range(2, 25):
    k = 1000
    TD, Med = build(data, k, distm)
    """
    med_list = [2304, 6435, 993, 5790, 2379, 11328, 181, 4461, 376, 2090,
                10137, 8858, 1826, 2953, 1831, 10090, 825, 1483, 4739, 5989,
                10136, 5313, 5112, 11293, 11229]
    print('total length med_list ', len(med_list))
    track_swap(data, med_list, 'runs/td-output.txt')
    """
    # data = data[0:12]

    # notes = '\t not beaking and continue.... \n'
    # notes = '\t breaking in the inner swap loop \n'
    # notes = '\tString of multiple runs to look at effect of k\n'
    # track(data, 25, file_name, notes)

    """
    for k in range(2, 20):
        start_time = dt.now()
        track(data, k, file_name, notes)
    """
