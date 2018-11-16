import pandas as pd
import matplotlib.pyplot as plt

# clustering imports
# import pyclustering.utils as pcu

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.kmedoids import kmedoids

from pyclustering.utils import read_sample
from pyclustering.utils import timedcall


# import data so all methods have access
df = pd.read_csv('monthly_avgs.csv', header=0, index_col=0)
attributes = df.columns.values


# x-y plot of two attributes
def plot2D(attribute_1, attribute_2):
    plt.plot(df[attributes[attribute_1]], df[attributes[attribute_2]], 'bo')
    plt.xlabel(attributes[attribute_1])
    plt.ylabel(attributes[attribute_2])
    plt.show()


# kmedoids template from example
def template_clustering(start_medoids, path, tolerance=0.25, show=True):
    sample = read_sample(path)

    kmedoids_instance = kmedoids(sample, start_medoids, tolerance)
    (ticks, result) = timedcall(kmedoids_instance.process)

    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    print("Sample: ", path, "\t\tExecution time: ", ticks, "\n")

    if show is True:
        visualizer = cluster_visualizer(1)
        visualizer.append_clusters(clusters, sample, 0)
        visualizer.append_cluster([sample[index] for index in start_medoids],
                                  marker='*', markersize=16)
        visualizer.append_cluster(medoids, data=sample, marker='*',
                                  markersize=15)
        visualizer.show()

    return (sample, clusters)


# show example with two attributes
def two_attribute_example():
    # start with 2 columns of data

    # write humidity and temperature to csv file
    df[[attributes[0], attributes[2]]].to_csv('h_t.csv', sep=' ', header=False,
                                              index=False)
    plot2D(0, 2)
    ex1 = template_clustering([3, 200], 'h_t.csv')
    print("Clusters for 3, 200")
    ex2 = template_clustering([20, 50], 'h_t.csv')


if __name__ == '__main__':
    print('running as main method')
    two_attribute_example()
