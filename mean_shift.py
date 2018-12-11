import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MeanShift
np.set_printoptions(precision=5, suppress=True)


start_time = datetime.now()

kernel_bandwidth = 100


def distance(x, xi):
	# return np.sqrt(np.sum((x-xi)**2))
	return np.linalg.norm(x-xi)


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
    d = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((distance / bandwidth))**2)
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

def mean_shift(data, columns):
	X = data[columns].copy()
	past_x = X.copy()
	X_orig = data[columns].copy()
	iterations = 10
	# past_X = []
	for it in range(iterations):
		print("ITERATION: ", it)
		print("Elapsed time:", datetime.now() - start_time)
		for i,x in X.iterrows(): #For each point in X
			# print("POINT ", i, ": ")
			X.iloc[i] = shift(X_orig,x)
			# print("x prime = ", X.iloc[i].values)
		# END LOOP
	# END LOOP
	return X
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
def mean_shift_vec(data, columns):
	X = data[columns].copy()
	shifted_pts = X.copy().values
	iterations = 1
	it = 0
	while True:
		it += 1
		print("ITERATION ", it)
		print("Elapsed time:", datetime.now() - start_time)
		pts_last = shifted_pts
		# Compute distance matrix for all points
		dist_matrix = euclidean_distances(shifted_pts,shifted_pts)

		# print("DIST MTX:")
		# print(dist_matrix)

		# Compute weights for all points
		weight_mtx = gauss_kernel(kernel_bandwidth, dist_matrix)

		# print("WEIGHT MTX:")
		# print(weight_mtx)

		exp_pts = np.dot(weight_mtx, shifted_pts)
		summed_weight = np.sum(weight_mtx, 0)
		# print("Exp pts:")
		# print(exp_pts)
		# print("Summed Weight: ")
		# print(summed_weight)
		shifted_pts = exp_pts / np.expand_dims(summed_weight, 1)

		diff = (shifted_pts - pts_last)**2
		diff = diff.sum(axis=-1)
		diff = np.sqrt(diff)
		# print("DIFF:")
		# print(diff)
		if np.all([j <= 0.01 for j in diff]): 
		# if it <= 5:
			break

	return shifted_pts

# END FUNCTION


# Test Version
def mean_shift_vec_test(data):
	X = data.copy()
	shifted_pts = X.copy().values
	iterations = 10

	# for it in range(iterations):
	i = 0
	while i < 5:
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
		print("WEIGHT MTX")
		print(weight_mtx)

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

data = pd.read_csv('./full-monthly-avgs.csv')
temp_data = data.iloc[:20]
print("Original data:")
for i,x in temp_data.iterrows():
	print(x[cols].values)


# test_data = pd.read_csv('./test-data/Simple12.csv', sep=' ', header=None)


print(MeanShift(bandwidth=kernel_bandwidth).fit(temp_data[cols]).labels_)
print(MeanShift(bandwidth=kernel_bandwidth).fit(temp_data[cols]).cluster_centers_)



shifted_pts = mean_shift(temp_data, cols)
final_df = pd.DataFrame(data=shifted_pts)
# print(final_df.drop_duplicates())

print("FINAL DATA:")
for i,row in final_df.iterrows():
	print(row.values)






print("RUN TIME: ", datetime.now() - start_time)

