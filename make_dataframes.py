import pandas as pd
from datetime import datetime

start_time = datetime.now()

def split_df():
	df_full = pd.read_csv('./get-weather-files/ws-cities-1-1000-data.csv', converters={'zip': lambda x: str(x)})
	df_1 = df_full[:1004500]
	df_2 = df_full[1004501:2009000]
	df_3 = df_full[2009001:3013500]
	df_4 = df_full[3013500:]

	print("Elapsed time: ", datetime.now() - start_time, ", read in all files")

	df_1.to_csv('./get-weather-files/ws-cities-1-250-data.csv')
	df_2.to_csv('./get-weather-files/ws-cities-250-500-data.csv')
	df_3.to_csv('./get-weather-files/ws-cities-500-750-data.csv')
	df_4.to_csv('./get-weather-files/ws-cities-750-1000-data.csv')

	print("Total run time: ", datetime.now() - start_time)

def join_dfs():
	df1 = pd.read_csv('./test-df-1-250.csv')
	df2 = pd.read_csv('./test-df-250-500.csv')
	df3 = pd.read_csv('./test-df-500-750.csv')
	df4 = pd.read_csv('./test-df-750-1000.csv')
	frames = [df1, df2, df3, df4]

	df_full = pd.concat(frames, ignore_index=True)
	print(df_full)
	df_full.to_csv('./full-monthly-avgs.csv')


join_dfs()

print("Run time:", datetime.now() - start_time)