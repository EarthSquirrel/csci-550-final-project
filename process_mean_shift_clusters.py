import pandas as pd
import numpy as np
np.set_printoptions(precision=5, suppress=True)

cols = ['PRCP','SNOW','SNWD','TMAX','TMIN']

def find_centers(df):
	X = df.copy()
	centers = X.round(3)
	centers = centers.drop_duplicates(subset=['PRCP','SNOW','SNWD','TMAX','TMIN'])
	return centers

# Write .txt file of cluster meta data including how many clusters
# and the the coords of the cluster centroids
def make_cluster_meta_file(df, bandwidth):
	file_name = 'bw_' + str(bandwidth) + '_conv_info.txt'
	file = open('./mean-shifted-data/'+file_name, 'w')
	file.write("Bandwidth = " + str(bandwidth) + "\n")

	centers = find_centers(df)
	file.write('Cluster count: '+ str(len(centers)) + '\n')
	print("Cluster count: ", len(centers))
	print("Centers:")
	file.write("Centers: \n")
	for i,c in centers.iterrows():
		print(c.values)
		file.write(str(c.values)+"\n")


# build set of clusters
def assign_clusters(clustered_df):
	centers = find_centers(clustered_df)
	indices = centers.index.values
	print("Indices = ", indices)
	# List of lists storing index values for each cluster
	clusters = [[] for x in range(len(centers))]
	# Get the actual center points from the clustered data
	center_pts = clustered_data.iloc[indices]
	print(center_pts)
	for i,x in clustered_df.iterrows():
		# print("x = ", x.values[1:])
		ci = 0
		for j,y in center_pts.iterrows():
			if np.array(x.values[1:] == y.values[1:]).all():
				# print("Point ", i, "is in cluster ", ci)
				clusters[ci].append(i)
				break
			ci += 1

	missing_pts = []
	# Find any points that weren't clustered
	for i in clustered_df.index:
		missing = True
		for c in clusters:
			if i in c:
				missing = False
				break
		if missing:
			missing_pts.append(i)
	print("Missing pts: ", missing_pts)
	# for pt in missing_pts:
	# 	for i,c in enumerate(clusters):
	# 		if np.array(clustered_df.iloc[c[0]].values[1:] == clustered_df.iloc[pt].values[1:]).all():
	# 			print("Adding point", pt, "to cluster")
	# 			c.append(pt)

	return(clusters)


def write_cluster_file(clusters, bandwidth):
	file = open("./mean-shifted-data/clusters_bw_"+str(bandwidth)+".csv", 'w')
	for c in clusters:
		for i in c:
			file.write(str(i)+",")
		file.write("\n")







orig_data = pd.read_csv('./monthly_avg_zscore.csv')
clustered_data = pd.read_csv('./mean-shifted-data/mean_shifted_data_bw_1.5_conv.csv')
clusters = assign_clusters(clustered_data)
write_cluster_file(clusters, 1.5)
make_cluster_meta_file(clustered_data, 1.5)


# X = clustered_data.copy()
# assign_clusters(orig_data, clustered_data, 1)





