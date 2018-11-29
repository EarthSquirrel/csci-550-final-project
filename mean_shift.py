import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
# from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import euclidean_distances

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
	return (1 / d)


def make_plot(data, columns, radius):
	filename = "../Mean Shift Trials/Points1-200/Radius" + str(radius) + ".png"
	fig,ax = plt.subplots()
	ax.plot(data[columns[0]].values, data[columns[1]].values, 'o')
	plt.savefig(filename)


def mean_shift(data, columns, radius):
	X = data[columns].copy()
	iterations = 5
	past_X = []
	break_loop = False
	for it in range(iterations):
		for i,x in X.iterrows(): #For each point in X
			#Find neighbors
			neighbors = neighborhood_pts(X, x, columns, radius)

			numerator = 0
			denominator = 0
			
			if len(neighbors) == 0: continue

			for neighbor in neighbors:
				dist = distance(neighbor[columns], x[columns])
				weight = gauss_kernel(kernel_bandwidth, dist)
				numerator += weight * neighbor[columns]
				denominator += weight
			# END LOOP
			x_prime = numerator / denominator
			# print("x prime: ", x_prime)
			# print("\n")
			X.loc[i, columns] = x_prime
		# END LOOP

		past_X.append(X.copy())
	# END LOOP
	return past_X
# END FUNCTION

# Vectorized attempt
def _mean_shift(data, columns):
	X = data[columns].copy()
# END FUNCTION




data = pd.read_csv('data/monthly_avgs.csv')
temp_data = data[['Temperature', 'Humidity']].iloc[0:3]



start_time = datetime.now()

# for radius in range(2,6):
# 	print("Radius = ", radius, "...")
# 	shifted_data = mean_shift(temp_data, ['Temperature', 'Humidity'], radius)
# 	shifted_df = pd.DataFrame(data=shifted_data[4])
# 	make_plot(shifted_df, ['Temperature', 'Humidity'], radius)

# print("RUN TIME: ", datetime.now() - start_time)

distance_mat = []
for i in temp_data.index:
	for j in temp_data.index:
		distance_mat.append(distance(temp_data.iloc[i], temp_data.iloc[j]))

distance_mat = np.array(distance_mat)
distance_mat.shape = (3,3)

print("VERSION 1:")
print(distance_mat)

# Convert dataframe to ndarray
arr = temp_data.values
# expd = np.expand_dims(arr, 2)
# tiled = np.tile(expd, arr.shape[0])
# trans = np.transpose(arr)
# diff = trans - tiled
# print(trans)
print("VERSION 2:")
print(euclidean_distances(arr, arr))















