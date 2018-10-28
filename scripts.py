import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import calendar
from threading import Thread
import multiprocessing as mp


#Create a list of columns we want from each table,
#so we don't have to specify it for each successive call to read_csv()
columns = ['datetime', 'Vancouver', 'Portland', 'San Francisco', 
		   'Seattle', 'Los Angeles', 'San Diego',
		   'Las Vegas', 'Phoenix', 'Albuquerque',
		   'Denver', 'San Antonio', 'Dallas',
		   'Houston', 'Kansas City', 'Minneapolis',
		   'Saint Louis', 'Chicago', 'Nashville',
		   'Indianapolis', 'Atlanta', 'Detroit',
		   'Jacksonville', 'Charlotte', 'Miami', 
		   'Pittsburgh', 'Toronto', 'Philadelphia',
		   'New York', 'Montreal', 'Boston']

#Build list of indices to drop
#We are dropping the few months in 2012 and 2017 
#so we have an even number of measurements for each month
drop_rows = list(range(0,2197)) + list(range(37260,45253))

humidity_df = pd.read_csv('data/humidity.csv', usecols = columns)
humidity_df = humidity_df.drop(humidity_df.index[drop_rows])
pressure_df = pd.read_csv('data/pressure.csv', usecols = columns)
pressure_df = pressure_df.drop(pressure_df.index[drop_rows])
temperature_df = pd.read_csv('data/temperature.csv', usecols = columns)
temperature_df = temperature_df.drop(temperature_df.index[drop_rows])
wind_direction_df = pd.read_csv('data/wind_direction.csv', usecols = columns)
wind_direction_df = wind_direction_df.drop(wind_direction_df.index[drop_rows])
wind_speed_df = pd.read_csv('data/wind_speed.csv', usecols = columns)
wind_speed_df = wind_speed_df.drop(wind_speed_df.index[drop_rows])


#Get a set of a city's monthly avgs for dataframe corresponding to a specific attribute
#param month int 1..12 for the corresponding month
#param dataframe one of the five dataframe corresponding to a weather attribute
#This is the original, slower function
def _get_monthly_avg(month, dataframe, city):
	#List of all values for the specificed month
	vals = []
	for index, row in dataframe.iterrows():
		date = pd.to_datetime(row['datetime'])
		if (date.month == month):
			#Use numpy's isnan function to check if current row is a missing value 
			#in which case append 0
			#TODO preprocess data to account for missing values
			if (np.isnan(row[city])):
				vals.append(0)
			else: 
				vals.append(row[city])
	
	return sum(vals) / len(vals)
#END FUNCTION

def get_monthly_avg(month, dataframe, city):

	vals = [] #Set of all monthy values of a specific weather attribute for one city
	for i in humidity_df.index:
		date = pd.to_datetime(dataframe.loc[i,'datetime'])
		if date.month == month: #If we are in the right month
			if (np.isnan(dataframe.loc[i, city])): #If current value is empty, just add 0
				vals.append(0)
			else: #Otherwise add current value to vals
				vals.append(dataframe.loc[i, city])
		else: #Once we've reached end of current month, jump ahead one year
			i += jump_one_year(date)

	return sum(vals) / len(vals)

def get_monthly_avgs(dataframe, city):
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
		date = pd.to_datetime(dataframe.loc[i, 'datetime'])
		if np.isnan(dataframe.loc[i, city]): continue
		if date.month == 1:
			jan_sum += dataframe.loc[i, city]
			jan_count += 1
		
		elif date.month == 2:
			feb_sum += dataframe.loc[i, city]
			feb_count += 1
		
		elif date.month == 3:
			mar_sum += dataframe.loc[i, city]
			mar_count += 1
		
		elif date.month == 4:
			apr_sum += dataframe.loc[i, city]
			apr_count += 1
		
		elif date.month == 5:
			may_sum += dataframe.loc[i, city]
			may_count += 1
		
		elif date.month == 6:
			jun_sum += dataframe.loc[i, city]
			jun_count += 1
		
		elif date.month == 7:
			jul_sum += dataframe.loc[i, city]
			jul_count += 1
		
		elif date.month == 8:
			aug_sum += dataframe.loc[i, city]
			aug_count += 1
		
		elif date.month == 9:
			sep_sum += dataframe.loc[i, city]
			sep_count += 1
		
		elif date.month == 10:
			octo_sum += dataframe.loc[i, city]
			octo_count += 1
		
		elif date.month == 11:
			nov_sum += dataframe.loc[i, city]
			nov_count += 1
		
		elif date.month == 12:
			dec_sum += dataframe.loc[i, city]
			dec_count += 1
	#END LOOP
	
	return [jan_sum / jan_count,
			feb_sum / feb_count,
			mar_sum / mar_count,
			apr_sum / apr_count,
			may_sum / mar_count,
			jun_sum / jun_count,
			jul_sum / jul_count,
			aug_sum / aug_count,
			sep_sum / sep_count,
			octo_sum / octo_count,
			nov_sum / nov_count,
			dec_sum / dec_count]






#Create a set of all monthly avgs of a specific attribute
#Set will become a column in the new dataframe
def make_column(dataframe):
	# cities = columns[1:] #columns list except for 'datetime' column
	cities = ['Vancouver','Portland']
	column = []
	for city in cities:
		print("Current city: ", city)
		column += [get_monthly_avg(i, dataframe, city) for i in range(1,13)]

	return column

def get_city_avgs(lock, city, dataframe, data_column):
	lock.acquire()
	print("Current city:", city)
	avgs = [get_monthly_avg(i, dataframe, city) for i in range(1,13)]
	print(city + " Avgs: " + str(avgs))
	data_column.append(avgs)
	lock.release()


#These functions are all used to jump ahead one year in the dataset
#Return set of all months in year, excluding the month parameter
def diff(month):
	months = set([i for i in range(1,13)])
	return [i for i in months if i != month]

#Get the number of days in a month, year is passed to account for leap years
def day_count(year, month):
	return calendar.monthrange(year, month)[1]

def jump_one_year(date):
	year = date.year
	month = date.month
	skip_months = diff(month)
	num_days = 0
	
	for m in skip_months:
		num_days += day_count(year, m)
		if m == 12:
			year += 1

	return num_days * 23
#END FUNCTION


# start_time = datetime.now()
# print(make_column(humidity_df))
# print(datetime.now() - start_time)

# humidity_col = mp.Queue()
# lock = mp.Lock()

# p1 = mp.Process(target=get_city_avgs, args=(lock, 'Vancouver', humidity_df, humidity_col))
# p2 = mp.Process(target=get_city_avgs, args=(lock, 'Portland', humidity_df, humidity_col))
# # t3 = mp.Process(target=get_city_avgs, args=['San Francisco', humidity_df, humidity_col])
# start_time = datetime.now()
# p1.start()
# p2.start()

# p1.join()
# p2.join()

# print(humidity_col.get())
# print(datetime.now() - start_time)
humidity_col = []
pressure_col = []
temperature_col = []
wind_direction_col = []
wind_speed_col = []
start_time = datetime.now()
for city in columns[1:]:
	print(city + ":")
	humidity_col.append(get_monthly_avgs(humidity_df, city))
	pressure_col.append(get_monthly_avgs(pressure_df, city))
	temperature_col.append(get_monthly_avgs(temperature_df, city))
	wind_direction_col.append(get_monthly_avgs(wind_direction_df, city))
	wind_speed_col.append(get_monthly_avgs(wind_speed_df, city))
print(datetime.now() - start_time)



	

