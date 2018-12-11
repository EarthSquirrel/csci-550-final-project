import pandas as pd
import numpy as np


monthly_avg = 'full-monthly-avgs.csv'

# read in csv file to df
df = pd.read_csv(monthly_avg, header=0, index_col=0)

columns = df.columns.values

for col in columns[2:]:
    raw = df[col]
    print(raw)
    mean = sum(raw)/len(raw)
    print("mean: ", mean)
    std = np.std(raw)
    print("std: ", std)
    z = []
    for x in raw:
        z.append((x-mean)/std)
    print("z avg: ", np.mean(z))
    # replace in df
    df[col] = z

# check it was replaced
print(df[:5])
df.to_csv('monthly_avg_zscore.csv')
