import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import calendar
from threading import Thread
import multiprocessing as mp

df_small = pd.read_csv('./get-weather-files/ws-cities-1-10-data.csv', converters={'zip': lambda x: str(x)})
df = pd.read_csv('./get-weather-files/ws-cities-1-1000-data.csv', converters={'zip': lambda x: str(x)})


def get_monthly_avgs(dataframe, zip_code, column):
	jan_sum = 0
	jan_count = 0
	feb_sum = 0
	feb_count = 0
	mar_sum = 0
	mar_count = 0
	apr_sum = 0
	apr_count = 0
	may_sum = 0
	may_count = 0
	jun_sum = 0
	jun_count = 0
	jul_sum = 0
	jul_count = 0
	aug_sum = 0
	aug_count = 0
	sep_sum = 0
	sep_count = 0
	octo_sum = 0
	octo_count = 0
	nov_sum = 0
	nov_count = 0
	dec_sum = 0
	dec_count = 0

	for i in dataframe.index:
		# Skip missing values
		if np.isnan(dataframe.loc[i,column]):
			continue
		# Skip over all other zip codes
		if dataframe.loc[i,'zip'] != zip_code: 
			continue
		
		month = dataframe.loc[i, 'month']
		
		if month == 1:
			jan_sum += dataframe.loc[i,column]
			jan_count += 1
		
		elif month == 2:
			feb_sum += dataframe.loc[i,column]
			feb_count += 1
		
		elif month == 3:
			mar_sum += dataframe.loc[i,column]
			mar_count += 1
		
		elif month == 4:
			apr_sum += dataframe.loc[i,column]
			apr_count += 1
		
		elif month == 5:
			may_sum += dataframe.loc[i,column]
			may_count += 1
		
		elif month == 6:
			jun_sum += dataframe.loc[i,column]
			jun_count += 1
		
		elif month == 7:
			jul_sum += dataframe.loc[i,column]
			jul_count += 1
		
		elif month == 8:
			aug_sum += dataframe.loc[i,column]
			aug_count += 1
		
		elif month == 9:
			sep_sum += dataframe.loc[i,column]
			sep_count += 1
		
		elif month == 10:
			octo_sum += dataframe.loc[i,column]
			octo_count += 1
		
		elif month == 11:
			nov_sum += dataframe.loc[i,column]
			nov_count += 1
		
		elif month == 12:
			dec_sum += dataframe.loc[i,column]
			dec_count += 1
	#END LOOP
	
	if jan_count == 0:
		jan = 0
	else: 
		jan = jan_sum / jan_count
	
	if feb_count == 0:
		feb = 0
	else:
		feb = feb_sum / feb_count
	
	if mar_count == 0:
		mar = 0
	else:
		mar = mar_sum / mar_count
	
	if apr_count == 0:
		apr = 0
	else:
		apr = apr_sum / apr_count
	
	if may_count == 0:
		may = 0
	else:
		may = may_sum / may_count
	
	if jun_count == 0:
		jun = 0
	else:
		jun = jun_sum / jun_count
	
	if jul_count == 0:
		jul = 0
	else:
		jul = jul_sum / jul_count
	
	if aug_count == 0:
		aug = 0
	else:
		aug = aug_sum / aug_count
	
	if sep_count == 0:
		sep = 0
	else:
		sep = sep_sum / sep_count
	
	if octo_count == 0:
		octo = 0
	else:
		octo = octo_sum / octo_count
	
	if nov_count == 0:
		nov = 0
	else:
		nov = nov_sum / nov_count
	
	if dec_count == 0:
		dec = 0
	else:
		dec = dec_sum / dec_count

	return [jan,feb,mar,apr,may,jun,jul,aug,sep,octo,nov,dec]

#END FUNCTION

def find_missing_vals(df):
	for i in df.index:




wind_speed_col = []
precip_col = []
snow_fall_col = []
snow_depth_col = []
max_temp_col = []
min_temp_col = []
smoke_col = []
high_wind_col = []

zip_codes = df.zip.unique()
columns = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WT08', 'WT11']

start_time = datetime.now()

i = 0
for zc in zip_codes:
	i += 1
	print("City 1/",len(zip_codes))
	wind_speed_col += get_monthly_avgs(df, zc, 'AWND')
	precip_col += get_monthly_avgs(df, zc, 'PRCP')
	snow_fall_col += get_monthly_avgs(df, zc, 'SNOW')
	snow_depth_col += get_monthly_avgs(df, zc, 'SNWD')
	max_temp_col += get_monthly_avgs(df, zc, 'TMAX')
	min_temp_col += get_monthly_avgs(df, zc, 'TMIN')
	smoke_col += get_monthly_avgs(df, zc, 'WT08')
	high_wind_col += get_monthly_avgs(df, zc, 'WT11')

d = {'WindSpeed': wind_speed_col,
	'Precipitation': precip_col,
	'SnowFall': snow_fall_col,
	'SnowDepth': snow_depth_col,
	'MaxTemp': max_temp_col,
	'MinTemp': min_temp_col,
	'Smoke': smoke_col,
	'HighWinds': high_wind_col}

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
indices = []
for zc in zip_codes:
	for month in months:
		indices.append(zc + '-' + month)

test_df = pd.DataFrame(data=d, index=indices)
test_df.to_csv('full_df_1.csv')

print("RUN TIME: ", datetime.now() - start_time)












