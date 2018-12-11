import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MeanShift
np.set_printoptions(precision=5, suppress=True)


start_time = datetime.now()

kernel_bandwidth = 0.75


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
    d = (1/(bandwidth * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((distance**2 / bandwidth**2)))
    return d


def make_neighbor_mtx(dist_mtx, radius):
	neighbor_mtx = np.zeros(dist_mat.shape)
	for i, row in enumerate(dist_mtx):
		for j, dist in enumerate(row):
			if dist <= 6:
				neighbor_mtx[i][j] = 1
	return neighbor_mtx


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

def shift(X, pt):
	numerator = 0
	denominator = 0
	for i,x in X.iterrows():
		# print("point", i, "=", x.values)
		dist = distance(x, pt)
		# print("dist = ", dist)
		weight = gauss_kernel(kernel_bandwidth, dist)
		# print("weight = ", weight)
		numerator += x * weight
		denominator += weight
	# END LOOP
	x_prime = numerator / denominator
	return x_prime

def mean_shift(data, columns, radius):
	X = data[columns].copy()
	past_x = X.copy()
	iterations = 1
	# past_X = []
	break_loop = False
	for it in range(iterations):
		dist_matrix = []
		print("ITERATION: ", it)
		print("Elapsed time:", datetime.now() - start_time)
		for i,x in X.iterrows(): #For each point in X
			print("POINT ", i, ": ")
			#Find neighbors
			# neighbors = neighborhood_pts(X, x, columns, radius)
			numerator = 0
			denominator = 0
			current_dist = []

			for j,y in X.iterrows():
				dist = np.linalg.norm(x[columns] - y[columns])
				current_dist.append(dist)
				weight = gauss_kernel(kernel_bandwidth, dist)
				numerator += weight * y[columns]
				denominator += weight
			# END LOOP
			dist_matrix.append(current_dist)
			print("numerator = ", numerator.values)
			print("denominator = ", denominator)
			x_prime = numerator.values / denominator
			print("xprime = ", x_prime)
			X.loc[i, columns] = x_prime
		# END LOOP
		past_x = X.copy()
	# END LOOP
	
	return past_x
# END FUNCTION

def mean_shift_test(data):
	X = data.copy()
	X_orig = data.copy()
	past_x = X.copy()
	iterations = 5
	# past_X = []
	break_loop = False
	for it in range(iterations):
		# print("ITERATION: ", it)
		# print("Elapsed time:", datetime.now() - start_time)
		for i,x in X.iterrows(): #For each point in X
			# print("POINT ", i, ": ")
			X.iloc[i] = shift(X_orig,x)
			# print("x prime = ", X.iloc[i].values)
		# END LOOP
	# END LOOP
	return X
# END FUNCTION



# Vectorized attempt
def mean_shift_vec(data, columns, radius):
	X = data[columns].copy()
	shifted_pts = X.copy()
	iterations = 1

	for it in range(iterations):
		print("ITERATION ", it)
		print("Elapsed time:", datetime.now() - start_time)
		
		# if it == 0:
		# 	dist_matrix = euclidean_distances(shifted_pts, shifted_pts)
		# else :
		# 	dist_matrix = euclidean_distances(shifted_pts,shifted_pts)
		dist_matrix = euclidean_distances(shifted_pts,shifted_pts)

		print("DIST MTX:")
		print(dist_matrix)

		weight_mtx = gauss_kernel(kernel_bandwidth, dist_matrix)

		print("WEIGHT MTX:")
		print(weight_mtx)

		exp_pts = np.dot(weight_mtx, shifted_pts)
		summed_weight = np.sum(weight_mtx, 0)
		print("Exp pts:")
		print(exp_pts)
		print("Summed Weight: ")
		print(summed_weight)
		shifted_pts = exp_pts / np.expand_dims(summed_weight, 1)
		# X = shifted_pts

	return shifted_pts

# END FUNCTION


# Test Version
def mean_shift_vec_test(data):
	X = data.copy()
	shifted_pts = X.copy().values
	iterations = 10

	# for it in range(iterations):
	i = 0
	while i<5:
		print("ITERATION ", i)
		i += 1
		print("Elapsed time:", datetime.now() - start_time)
		pts_last = shifted_pts
		
		dist_matrix = euclidean_distances(shifted_pts, shifted_pts)
		# print("DIST MTX:")
		# print(dist_matrix)
		# print("PTS LAST:")
		# print(pts_last)

		weight_mtx = gauss_kernel(kernel_bandwidth, dist_matrix)
		# print("WEIGHT MTX")
		# print(weight_mtx)

		exp_pts = np.dot(weight_mtx, shifted_pts)
		
		# print("EXPD PTS")
		# print(exp_pts)

		summed_weight = np.sum(weight_mtx, 0)

		# print("Summed Weight:")
		# print(summed_weight)

		shifted_pts = exp_pts / np.expand_dims(summed_weight, 1)

		# print("SHIFTED PTS:")
		# print(shifted_pts)

		# diff = (shifted_pts - pts_last)**2
		# diff = diff.sum(axis=-1)
		# diff = np.sqrt(diff)
		# print("DIFF:")
		# print(diff)
		# if np.all([j <= 0.0001 for j in diff]): 
		# 	break

	print("Total iterations:",i)
	return shifted_pts

# END FUNCTION


cols = ['PRCP','SNOW','SNWD','TMAX','TMIN']

# data = pd.read_csv('./full-monthly-avgs.csv')
# temp_data = data.iloc[0:10]


test_data = pd.read_csv('./test-data/Simple12.csv', sep=' ', header=None)


# for i,x in X.iterrows():
# 	print("i = ", i)
# 	print("x = ", x)
# 	for j,y in X.iterrows():
# 		print("j = ", j)
# 		print("y = ", y)

print(MeanShift(bandwidth=1).fit(test_data).labels_)
print(MeanShift(bandwidth=1).fit(test_data).cluster_centers_)



shifted_pts = mean_shift_vec_test(test_data)
print(shifted_pts)





print("RUN TIME: ", datetime.now() - start_time)

