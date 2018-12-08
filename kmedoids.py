import math
import pandas as pd
import numpy as np


# read in data set
def read_data(csv_name):
    # read in from pandas, then convert to matrix
    data = pd.read_csv(csv_name).values
    return data


# compute distance function
def calc_dist(x1, x2):
    Z = x1-x2
    dist = np.sqrt(sum([z*z for z in Z]))
    print('distance: {}\n\t for: {}'.format(dist, Z))
    return dist


def build(data, k):
    # initialize variables
    TD = math.inf  # total deviation
    M = [(math.inf, math.inf)]  # set of medoids

    # find first medoid
    for i, xi in enumerate(data):
        tdj = 0
        # itterate through all data points that are not equal to xi
        for o in [ii for ii in range(len(data)) if ii != i]:
            tdj = tdj + calc_dist(xi, data[o])
            if tdj < TD:
                TD, M[0] = tdj, xi
                print('assigned new TD: {} for xi: {}'.format(TD, i))


def swap(data, TD, M):
    # TODO: Put code for swap method here
    pass


if __name__ == '__main__':
    print('Running as main class')
    k = 2  # number of clusters
    data = read_data('simple01.csv')
    build(data, k)
