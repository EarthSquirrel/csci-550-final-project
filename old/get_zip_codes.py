import pandas as pd
import numpy as np

# preset number cities and output file name
num_sample_cities = 500
output_csv = 'list-500-cities.csv'

raw = pd.read_csv('uszipsv1.4.csv')

# lists to hold condenced down values
zip_codes = []
city = []
state = []

# keep track of current values
cur_city = raw['city'][0]
zip_list = []

# itterate through list and condence down based on city
for x in range(0, len(raw['city'])):
    if raw['city'][x] == cur_city:
        zip_list.append(raw['zip'][x])
    else:
        zip_codes.append(zip_list)
        city.append(cur_city)
        state.append(raw['state_id'][x])
        # If not at the end of the list, assing new values
        if x < len(raw['city']):
            cur_city = raw['city'][(x + 1)]
            zip_list = []

# pick 500 random cities & extra 20 to account for null values
city_indices = np.random.choice(len(city), num_sample_cities,
                                replace=False)

# Pick the final citites and put into dataframe
final_city, final_zip, final_state = [], [], []
for i in city_indices:
    # assume that all rows have a zip code, but if not, size will be smaller
    if len(zip_codes[i]) > 0:
        final_city.append(city[i])
        final_zip.append(zip_codes[i][np.random.choice(
                         len(zip_codes[i]), 1)[0]])
        final_state.append(state[i])

# create final data frame for citites list
final_cities = pd.DataFrame(data={'zip': final_zip, 'city': final_city,
                                  'state': final_state})

print("Length of final citites is: {}".format(len(final_cities['zip'])))

# write final citites to a csv
final_cities.to_csv(output_csv, index=False)
