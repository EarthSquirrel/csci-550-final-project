import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import calendar
from threading import Thread
import multiprocessing as mp
import concurrent.futures
from itertools import repeat

start_time = datetime.now()
# df = pd.read_csv('./get-weather-files/ws-cities-1-10-data.csv', converters={'zip': lambda x: str(x)})
# df = pd.read_csv('./get-weather-files/ws-cities-1-5-data.csv', converters={'zip': lambda x: str(x)})
df = pd.read_csv('./get-weather-files/ws-cities-5-10-data.csv', converters={'zip': lambda x: str(x)})

# df_1 = pd.read_csv('./get-weather-files/ws-cities-1-250-data.csv', converters={'zip': lambda x: str(x)})
# df_2 = pd.read_csv('./get-weather-files/ws-cities-250-500-data.csv', converters={'zip': lambda x: str(x)})
print("Read all files")

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


def _get_monthly_avgs(df,zip_code):
	cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WT08', 'WT11']

	jan_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	jan_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	feb_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	feb_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})  
	mar_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	mar_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	apr_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	apr_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	may_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	may_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	jun_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	jun_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	jul_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	jul_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	aug_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	aug_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	sep_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	sep_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	oct_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	oct_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	nov_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	nov_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	dec_sums = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0})
	dec_counts = dict({'AWND' : 0, 'PRCP' : 0, 'SNOW' : 0, 'SNWD' : 0, 'TMAX' : 0, 'TMIN' : 0, 'WT08' : 0, 'WT11' : 0}) 
	
	for i in df.index:
		if df.loc[i,'zip'] != zip_code:
			continue
		month = df.loc[i,'month']
		if month == 1:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					jan_sums[col] += df.loc[i,col]
					jan_counts[col] += 1
		
		elif month == 2:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					feb_sums[col] += df.loc[i,col]
					feb_counts[col] += 1
		
		elif month == 3:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					mar_sums[col] += df.loc[i,col]
					mar_counts[col] += 1

		elif month == 4:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					apr_sums[col] += df.loc[i,col]
					apr_counts[col] += 1

		elif month == 5:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					may_sums[col] += df.loc[i,col]
					may_counts[col] += 1

		elif month == 6:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					jun_sums[col] += df.loc[i,col]
					jun_counts[col] += 1

		elif month == 7:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					jul_sums[col] += df.loc[i,col]
					jul_counts[col] += 1

		elif month == 8:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					aug_sums[col] += df.loc[i,col]
					aug_counts[col] += 1

		elif month == 9:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					sep_sums[col] += df.loc[i,col]
					sep_counts[col] += 1

		elif month == 10:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					oct_sums[col] += df.loc[i,col]
					oct_counts[col] += 1

		elif month == 11:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					nov_sums[col] += df.loc[i,col]
					nov_counts[col] += 1

		elif month == 12:
			for col in cols:
				if np.isnan(df.loc[i,col]):
					continue
				else:
					dec_sums[col] += df.loc[i,col]
					dec_counts[col] += 1
	#END LOOP

	jan_avgs = [0 if jan_counts[col] == 0 else (jan_sums[col] / jan_counts[col]) for col in cols]
	feb_avgs = [0 if feb_counts[col] == 0 else (feb_sums[col] / feb_counts[col]) for col in cols]
	mar_avgs = [0 if mar_counts[col] == 0 else (mar_sums[col] / mar_counts[col]) for col in cols]
	apr_avgs = [0 if apr_counts[col] == 0 else (apr_sums[col] / apr_counts[col]) for col in cols]
	may_avgs = [0 if may_counts[col] == 0 else (may_sums[col] / may_counts[col]) for col in cols]
	jun_avgs = [0 if jun_counts[col] == 0 else (jun_sums[col] / jun_counts[col]) for col in cols]
	jul_avgs = [0 if jul_counts[col] == 0 else (jul_sums[col] / jul_counts[col]) for col in cols]
	aug_avgs = [0 if aug_counts[col] == 0 else (aug_sums[col] / aug_counts[col]) for col in cols]
	sep_avgs = [0 if sep_counts[col] == 0 else (sep_sums[col] / sep_counts[col]) for col in cols]
	oct_avgs = [0 if oct_counts[col] == 0 else (oct_sums[col] / oct_counts[col]) for col in cols]
	nov_avgs = [0 if nov_counts[col] == 0 else (nov_sums[col] / nov_counts[col]) for col in cols]
	dec_avgs = [0 if dec_counts[col] == 0 else (dec_sums[col] / dec_counts[col]) for col in cols]
		
	return [jan_avgs, feb_avgs, mar_avgs, 
			apr_avgs, may_avgs, jun_avgs, 
			jul_avgs, aug_avgs, sep_avgs, 
			oct_avgs, nov_avgs, dec_avgs]

# END FUNCTION


def _make_avgs_df(df):
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

	i = 0
	for zc in zip_codes:
		i += 1
		print("Zip ", i, "/",len(zip_codes))
		print("Elapsed time: ", datetime.now() - start_time)
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

# END FUNCTION

def make_avgs_df(df):
	zip_codes = df.zip.unique()
	cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WT08', 'WT11']
	avgs = []
	i = 0
	for zc in zip_codes:
		i += 1
		print("On zip", i, "/", len(zip_codes))
		avgs += _get_monthly_avgs(df,zc)

	return avgs

def build_dataframe(df1, df2):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		frames = [df1, df2]
		avgs = executor.map(make_avgs_df, frames)

	avgs = list(avgs)[0]
	
	print(len(avgs))
	
	cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WT08', 'WT11']
	months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	indices = []
	
	zip_codes = list(df1.zip.unique())
	zip_codes += list(df2.zip.unique())

	for zc in zip_codes:
		for month in months:
			indices.append(zc + '-' + month)

	monthly_avg_df = pd.DataFrame(columns=cols, index=indices)

	print(len(monthly_avg_df))

	# for i in range(len(monthly_avg_df)):
	# 	monthly_avg_df.loc[indices[i]] = avgs[i]

	# print(monthly_avg_df)

	# monthly_avg_df.to_csv('./test-df-1-10.csv')
	


def get_zip_counts(zip_code):
	count = 0
	for i in df_1.index:
		if df_1.loc[i,'zip'] == zip_code:
			count += 1

	return count

# END FUNCTION


zip_codes = df.zip.unique()
cols = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'WT08', 'WT11']

avgs = []

i = 0
for zc in zip_codes:
	i += 1
	print(i, "/", len(zip_codes))
	print("Elapsed time:", datetime.now() - start_time)
	avgs += _get_monthly_avgs(df, zc)


months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
indices = []

for zc in zip_codes:
	for month in months:
	    indices.append(zc + '-' + month)


monthly_avg_df = pd.DataFrame(columns=cols, index=indices)

for i in range(len(monthly_avg_df)):
        monthly_avg_df.loc[indices[i]] = avgs[i]

print(monthly_avg_df)
monthly_avg_df.to_csv('./test-df-5-10.csv')



print("RUN TIME: ", datetime.now() - start_time)


