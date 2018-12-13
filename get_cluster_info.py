# analyze cluster information
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


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
            if len(line) > 0:
                line = line.strip()
            try:
                # print('line 0: ', line[0])
                int(line[0])
                if line[-1] == ',':
                    line = line[:-2]
                    # print('new line end: ', line[-1])
                temp = list(map(int, line.strip().split(',')))
                single_group.append(temp)
                # print('len single group ', len(single_group))
            except (ValueError, IndexError):
                # print('Value error: ', line[0])
                # start of new cluster group, start new group
                if len(single_group) > 0:
                    all_clusters.append(single_group)
                    single_group = []

            line = f.readline()

    print('len all {} clusters'.format(len(all_clusters)))
    for g in all_clusters:
        # print(g)
        print('\t', len(g))

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
    zips, city, state, months_per_city = [], [], [], []
    for uzip in unique_zip:
        loc = zip_list.index(uzip)
        months_per_city.append([])
        zips.append(cluster_df['zip'][loc])
        city.append(cluster_df['city'][loc])
        state.append(cluster_df['state'][loc])

    # append the months onto correct city
    for i, x in cluster_df.iterrows():
        index = unique_zip.index(x['zip'])
        months_per_city[index].append(x['month'])

    # for c in months_per_city:
    #     print('\t{} months in {}'.format(len(c)-3, c[1]))

    d = {'zip': zips, 'city': city, 'state': state, 'months': months_per_city}
    months_per_city = pd.DataFrame(data=d)

    return months_per_city


# take months per city and return array of values
def month_to_number_array(mpc, num_colors=5):
    # create empty array for months cities X 12
    num_values = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec']
    for i, row in mpc.iterrows():
        value = i % num_colors + 1.5
        # print('filling with: ', value, ' for ', i)
        temp = [.5 for dummy in range(12)]
        # get the months and fill temp
        for m in row['months']:
            temp[months.index(m)] = value
        # print(row['months'])
        # print('numerical months: ', temp)
        num_values.append(temp)

    return num_values


# take in an
def plot_color_city(months_per_city):
    scale = 3
    # get the matrix for the colors
    numerical_months = month_to_number_array(months_per_city)
    if len(numerical_months) > 50:
        numerical_months = numerical_months[0:49]
        numerical_months.append([8 for x in range(12)])
    # print(numerical_months)

    # partial code from example
    # create discrete colormap
    cmap = colors.ListedColormap(['w', 'r', 'b', 'g', 'orange', 'purple',
                                  'black'])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]

    num2 = []
    # increase by 10
    for i in range(len(numerical_months)):
        nm = []
        for j in range(len(numerical_months[i])):
            for tw in range(scale):
                nm.append(numerical_months[i][j])

        num2.append(nm)

    numerical_months = num2.copy()
    norm = colors.BoundaryNorm(bounds, cmap.N)


    fig, ax = plt.subplots()
    ax.imshow(numerical_months, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='x', linestyle='-', color='k', linewidth=1)
    # ax.set_yticks(np.arange(-.5, 10, 1));
    x_value = [i*scale-.5 for i in range(0,12)]
    ax.set_xticks(x_value)

    # plot it
    plt.show()


def readable_string(city_month):
    csi = '\x1B['
    red = csi + '31;1m'
    yellow = csi + '33;1m'
    end = csi + '0m'
    final = ''
    for i, cm in city_month.iterrows():
        final += '{}: {}, {}\n'.format(cm['zip'], cm['city'],
                                       cm['state'])
    return final


if __name__ == '__main__':
    print('running as main')
    # read in data as df
    data = pd.read_csv('monthly_avg_zscore.csv')
    print(data.columns.values)

    all_clusters = read_clusters('means-cluster-15.txt')
    print('len of all clusters: ', len(all_clusters))
    for i, a in enumerate(all_clusters):
        print('num of clusters: ', len(a))
        print('\tindex of ', i)
        for aa in a:
            print('\t\tnumber of values in clusters: ', len(aa))

    test = all_clusters[0]
    for t in test:
        print('len t in test: ', len(t))
    # print('test: ', test)

    # *************************************
    # this is a set of clusters
    print('len t: ', len(t))
    dfc = get_city_state(test, data)
    print('len t {} len city/state {}'.format(len(t), len(dfc)))

    for i, c in enumerate(dfc):
        # ********************************
        # this must be done by each cluster
        city_month = sort_by_city(c)
        out = '*'.join(['*****' for x in range(10)]) + '\n'
        out += 'Cluster {} of length {} \n'.format(i, len(c))

        # get readablel ist
        out += readable_string(city_month) + '\n\n'
        with open('figures-info.txt', 'a') as f:
            f.write(out)

        plot_color_city(city_month)

    """
    for c in dfc:
        city_months = sort_by_city(c)
    # for r in city_months:
        # print(r)

        plot_color_city(city_months)
        break

    for t in test:
        dfc = get_city_state(t, data)
        print('len t {} len city/state {}'.format(len(t), len(dfc)))
        for c in dfc:
            city_months = sort_by_city(c)
    # for r in city_months:
        # print(r)

            plot_color_city(city_months)
            break
    """

    """
    all_clusters = read_clusters('clusters.txt')
    print(len(all_clusters))
    for i, a in enumerate(all_clusters):
        print('num of clusters: ', len(a))
        print('\tindex of ', i)
        for aa in a:
            print('\t\tnumber of values in clusters: ', len(aa))

    test = all_clusters[13]
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

            plot_color_city(city_months)
            break
    """
