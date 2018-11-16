import pandas as pd
import numpy as np
import math
# import matplotlib.pyplot as plt
from datetime import datetime
# from sklearn.neighbors import BallTree

kernel_bandwidth = 4


def distance(x, xi):
	return np.sqrt(np.sum((x-xi)**2))


def neighborhood_pts(X, x_centroid, columns, radius=6):
	neighbors = []
	
	for i,row in X.iterrows():
		#Skip don't add x_centroid to the neighborhood
		if row.name == x_centroid.name: continue
		# print(row.name,distance(row[columns], x_centroid[columns]))
		if distance(row[columns], x_centroid[columns]) <= radius:
			neighbors.append(row)
		
	return neighbors


def gauss_kernel(bandwidth, distance):
	d = (bandwidth * np.sqrt(2 * math.pi) * np.exp(-0.5 * ((distance / bandwidth)**2)))
	print(d)
	return (1 / d)

#TODO pass columns array as a parameter
def mean_shift(data, columns):
	X = data[columns].copy()
	iterations = 5
	past_X = []
	break_loop = False
	for it in range(iterations):
		for i,x in X.iterrows():
			#Find neighbors
			neighbors = neighborhood_pts(X, x, columns)

			numerator = 0
			denominator = 0
			
			if len(neighbors) == 0: continue

			for neighbor in neighbors:
				print("neighbor : ", neighbor)
				print("x : ", x)
				dist = distance(neighbor[columns], x[columns])
				weight = gauss_kernel(kernel_bandwidth, dist)
				print("Distance :", dist)
				print("Weight: ", weight)
				numerator += weight * neighbor[columns]
				denominator += weight
				print("Numerator: ", numerator)
				print("Denominator: ", denominator)
				print("\n")
			
			x_prime = numerator / denominator
			print("x prime: ", x_prime)
			print("\n")
			X.loc[i, columns] = x_prime

		past_X.append(X.copy())

	print(past_X)


data = pd.read_csv('data/monthly_avgs.csv')
temp_data = data.iloc[0:30]



start_time = datetime.now()
mean_shift(data, ['Temperature', 'Humidity'])
print("RUN TIME: ", datetime.now() - start_time)









