import pandas as pd
from datetime import datetime

start_time = datetime.now()

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