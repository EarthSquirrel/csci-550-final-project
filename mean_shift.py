import pandas as pd
import numpy as np
import math

kernel_bandwidth = 4


def distance(x, xi):
	return np.sqrt(np.sum((x-xi)**2))


def neighborhood_pts(X, x_centroid, radius=5):
	neighbors = []
	
	for i,row in X.iterrows():
		#Skip don't add x_centroid to the neighborhood
		if row.name == x_centroid.name: continue
		if distance(row, x_centroid) <= radius:
			neighbors.append(row)
		

	return neighbors


def gauss_kernel(bandwidth, distance):
	return (1 / (bandwidth * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((distance / bandwidth)**2))

#TODO pass columns array as a parameter
def mean_shift(data, column):
	X = data.copy()
	iterations = 5
	past_X = []
	break_loop = False
	for it in range(iterations):
		for i,x in X.iterrows():
			#Find neighbors
			neighbors = neighborhood_pts(X, x)

			#Find m(x) for each x in neighbors
			numerator = 0
			denominator = 0
			for xi in neighbors:
				dist = distance(xi[column], x[column])
				print("xi : ", xi[column])
				print("x : ", x[column])
				print("dist : ", dist)
				weight = gauss_kernel(dist, kernel_bandwidth) #Find K(xi - x)
				print("weight : ", weight)
				numerator += weight * xi[column]
				denominator += weight
				print("numerator : ", numerator)
				print("denominator : ", denominator)
			x_prime = numerator / denominator
			print("x_prime : ", x_prime)
			X.loc[i, column] = x_prime

		past_X.append(X.copy())

	print(past_X)


data = pd.read_csv('data/monthly_avgs.csv')
temp_data = data[['Temperature'].iloc[1:30]]

mean_shift(temp_data, 'Temperature')
# dist = distance(temp_data.iloc[0], temp_data.iloc[1])
# print(temp_data.iloc[7])
# neighbors = neighborhood_pts(temp_data, temp_data.iloc[7])
# for n in neighbors:
# 	print(n)








