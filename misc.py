import pandas as pd


# make global variables
zip_info_file = 'list-1000-cities.csv'
zip_info = pd.read_csv(zip_info_file, converters={'zip': lambda x: str(x)})
list_of_zips = list(zip_info['zip'])
# zip_info.set_index('zip')
print(zip_info.columns.values)


def get_cities_in_data(data_df):
    # get unique zip codes, separate from months
    zip_mon = data_df['zip_month']
    zip_list = [zm.split('-')[0] for zm in zip_mon]
    uni_zip = list(set(zip_list))
    print('len comparisn:: zips {} unique {}'.format(len(zip_list),
                                                     len(uni_zip)))

    # find cities corrisponding to zip codeas and put in list
    list_cities = []
    for uz in uni_zip:
        loc = list_of_zips.index(uz)
        info = zip_info['city'][loc] + ', ' + zip_info['state'][loc]
        info = [zip_info['city'][loc], zip_info['state'][loc], uz]
        # print(info)
        list_cities.append(info)

    return list_cities


def print_col_as_r(r_file, file_name):
    df = pd.read_csv(file_name, header=None)
    out = '\n'
    for i in range(len(df)):
        out += ', ' + str(round(df[1][i], 4))

    print(out)

    with open(r_file, 'a') as f:
        f.write(out)
        f.write('\n\n')


if __name__ == '__main__':
    print('running as main')

    # read in data matrix as df
    data_df = pd.read_csv('monthly_avg_zscore.csv')
    all_cities = get_cities_in_data(data_df)

    # find number of states
    states = [cs[1] for cs in all_cities]
    print('total states: ', len(set(states)))

    all_cities = [[ac[1], ac[0], ac[2]] for ac in all_cities]
    # sort by states
    all_cities.sort()

    output = ''
    for x in all_cities:
        output += '{}: {}, {}\n'.format(x[2], x[1], x[0])

    with open('list-included-cities.txt', 'w') as f:
        f.write(output)
