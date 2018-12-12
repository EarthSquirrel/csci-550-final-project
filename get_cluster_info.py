# analyze cluster information
import pandas as pd


# make global variables
zip_info_file = 'list-1000-cities.csv'
zip_info = pd.read_csv(zip_info_file, converters={'zip': lambda x: str(x)})
# zip_info.set_index('zip')
print(zip_info.columns.values)


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


# only takes one group of clusters
def get_city_state(clusters, data):
    output = []
    # itterate through each cluster
    zip_list = list(zip_info['zip'])
    for cluster in clusters:
        # cluster_num, zip_value, month, city, state = [], [], [], [], []
        info = [[], [], [], []]
        for c in cluster:
            # find value in clusters
            zip_month = data['zip_month'][c].split('-')

            # look for zip code and print error if did not find
            try:
                loc = zip_list.index(zip_month[0])
                # add values to appropriate lists
                values = [zip_month[0], zip_month[1], zip_info['city'][loc],
                          zip_info['state'][loc]]
                for j, v in enumerate(values):
                    info[j].append(v)

            except ValueError:
                print('*** Problem with cluster: {}'.format(c))
                print('\t{} was not found list {}'.format(zip_month[0],
                      zip_info_file))
        d = {'zip': info[0], 'month': info[1], 'city': info[2],
             'state': info[3]}
        output.append(pd.DataFrame(data=d))

        # return: each cluster is a dataframe
    return output


# finds the unique cities in the cluster and what months for each
def sort_by_city(cluster_df):
    # find number of unique cities by zip codes
    unique_zip = list(cluster_df['zip'].unique())
    print('length of unique: ', len(unique_zip))

    # find all the months for each city
    months_per_city = []
    zip_list = list(cluster_df['zip'])
    for uzip in unique_zip:
        loc = zip_list.index(uzip)
        months_per_city.append([cluster_df['zip'][loc],
                               cluster_df['city'][loc],
                               cluster_df['state'][loc]])

    # append the months onto correct city
    for i, x in cluster_df.iterrows():
        index = unique_zip.index(x['zip'])
        months_per_city[index].append(x['month'])

    for c in months_per_city:
        print('\t{} months in {}'.format(len(c)-3, c[1]))
        pass

    return months_per_city


if __name__ == '__main__':
    print('running as main')
    # read in data as df
    data = pd.read_csv('monthly_avg_zscore.csv')
    print(data.columns.values)

    all_clusters = read_clusters('clusters.txt')
    print(len(all_clusters))
    for a in all_clusters:
        print('num of clusters: ', len(a))

    test = all_clusters[0:2]
    for t in test:
        print('len t in test: ', len(t))
        for tt in t:
            print('\tlen tt in t: ', len(tt))
    # print('test: ', test)

    for t in test:
        dfc = get_city_state(t, data)
        print('len t {} len city/state {}'.format(len(t), len(dfc)))
        for c in dfc:
            city_months = sort_by_city(c)
    # for r in city_months:
        # print(r)
