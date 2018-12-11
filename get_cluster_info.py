# analyze cluster information
import pandas as pd


# make global variables
zip_info_file = 'list-1000-cities.csv'


# read in file, and ignore comments  with mark #
def read_clusters(clusters_file):
    # make list to hold each set of clusters
    # *** Clusters groups must be separated something that isn't a number
    all_clusters = []
    with open(clusters_file) as f:
        line = f.readline()
        single_group = []
        while line:
            try:
                int(line[0])
                temp = list(map(int, line.strip().split(',')))
                single_group.append(temp)
            except ValueError:
                # start of new cluster group, start new group
                if len(single_group) > 0:
                    all_clusters.append(single_group)
                    single_group = []

            line = f.readline()

    # for g in all_clusters:
        # print(len(g))

    return all_clusters


def get_city_state(cluster):
    # TODO: Get city and state from list 1000-cities
    pass


if __name__ == '__main__':
    print('running as main')
    read_clusters('clusters.txt')
