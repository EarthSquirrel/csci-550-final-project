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

# Find neighbors using distance matrix
def get_neighbors(index, dist_mat, radius=6):
	distances = dist_mat[index]
	neighbors = []
	for i in range(len(distances)):
		if distances[i] < 6:
			neighbors.append(temp_data.iloc[i])
	return neighbors


def gauss_kernel(bandwidth, distance):
	d = (bandwidth * np.sqrt(2 * math.pi) * np.exp(-0.5 * ((distance / bandwidth)**2)))
	return (1 / d)


def make_neighbor_mtx(dist_mtx, radius):
	neighbor_mtx = np.zeros(dist_mat.shape)
	for i, row in enumerate(dist_mtx):
		for j, dist in enumerate(row):
			if dist <= 6:
				neighbor_mtx[i][j] = 1
	return(neighbor_mtx)


def get_neighbors_from_mtx(point_index, neighbor_mtx, data):
	neighbors = []
	for i,val in enumerate(neighbor_mtx[point_index]):
		if val == 1:
			neighbors.append(data.iloc[i])

	return neighbors


def make_plot(data, columns, radius):
	filename = "../Mean Shift Trials/Full_Data_Set" + str(radius) + ".png"
	fig,ax = plt.subplots()
	ax.plot(data[columns[0]].values, data[columns[1]].values, 'o')
	plt.savefig(filename)


def mean_shift(data, columns, radius):
	X = data[columns].copy()
	iterations = 5
	past_X = []
	break_loop = False
	for it in range(iterations):
		print("ITERATION: ", it)
		for i,x in X.iterrows(): #For each point in X
			print("Point ", i, "/", len(X))
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
			X.loc[i, columns] = x_prime
		# END LOOP
		past_X.append(X.copy())
	# END LOOP
	return past_X
# END FUNCTION

# Vectorized attempt
def mean_shift_vec(data, columns, radius):
	X = data[columns].copy()
	# build matix of distances
	# Each row stores an array of distances to every other point from that point
	# I.e. row 1 stores distances to each other point from point 1
	dist_matrix = euclidean_distances(X.values. X.values)
	weight_mtx = gauss_kernel(kernel_bandwidth, dist_matrix)

	for i,row in X.iterrows():
		numerator = np.sum(weight_mtx[i] * X.values[i])
		denominator = np.sum(weight_mtx[i])
		x_prime = numerator / denominator

# END FUNCTION




data = pd.read_csv('data/monthly_avgs.csv')
temp_data = data[['Temperature', 'Humidity']].iloc[0:10]


start_time = datetime.now()
shifted_data = mean_shift(data, ['Temperature', 'Humidity'], 5)
shifted_df = pd.DataFrame(data=shifted_data[4])
make_plot(shifted_df, ['Temperature', 'Humidity'], 5)
print(shifted_df)
shifted_df.to_csv('../Mean Shift Trials/shifted_data.csv')
print("RUN TIME: ", datetime.now() - start_time)



# start_time = datetime.now()
# arr = temp_data.values
# dist_mat = euclidean_distances(arr, arr)
# weight_mtx = gauss_kernel(kernel_bandwidth, dist_mat)
# neighbor_mtx = make_neighbor_mtx(dist_mat, 6)
# mean_shift(temp_data, ['Temperature', 'Humidity'], 6)























