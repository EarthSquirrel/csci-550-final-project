import pandas as pd

df = pd.read_csv('cut-monthly-avgs.csv')

endRow = []

for i, r in df.iterrows():
    if r['TMAX'] != 0 or r['TMIN'] != 0:
        endRow.append(i)

cut = df.iloc[endRow, 0:]
print(cut.columns.values)
print(len(cut))
print(len(endRow))

cut.to_csv('cut-monthly-avgs.csv')
